"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Network definition for our shape and appearance encoder model.
Heavily inspired by the network architecture described in https://arxiv.org/pdf/1903.06946.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional
from collections import namedtuple
from .submodules import estimate_gaussian_params, gaussian_params_to_heatmap
from .generator import SPADEGenerator
from .unet import Unet


class PartFactorizedModel(nn.Module):
    def __init__(self, args):
        super(PartFactorizedModel, self).__init__()
        self.n_landmarks = args['n_landmarks']
        appearance_feat_dim = args['n_filters']
        # half_res_out means output is half resolution of input
        self.downsampler = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.use_identity_covariance = args['use_identity_covariance']
        self.fixed_covar = args['fixed_covar']
        self.img_size = args['img_size']
        self.use_fg_bg_mask = args['use_fg_bg_mask']
        self.shape_encoder = Unet(3, [512, 256, 128, self.n_landmarks], 4, 3, args['nsf'])
        additional_filters = self.shape_encoder.f_filters
        self.appearance_encoder = Unet(self.n_landmarks + additional_filters,
                                       [64, appearance_feat_dim], 1, 1, args['naf'])
        # setup generator opts for SPADE
        SPADEOptions = namedtuple('Options', ['ngf', 'appearance_nc', 'semantic_nc', 'norm_G'])
        opt = SPADEOptions(ngf=args['ngc'], appearance_nc=appearance_feat_dim,
                           semantic_nc=self.n_landmarks, norm_G='spectralspadeinstance3x3')
        self.image_decoder = SPADEGenerator(opt, self.img_size, self.img_size)

        if self.use_fg_bg_mask: # use foreground background masking
            print("Using foreground-background masking!")
            self.bg_net = Unet(3, [32, 64, 128, 3], 3, 3, 32)
            mask_outdim = 1
            if args['low_res_mask']:  # use a lower resolution mask to avoid encoding hifreq detail 
                self.fg_mask_net = nn.Sequential(Unet(self.n_landmarks, [32, mask_outdim], 3, 1, 32),
                                                 nn.Upsample(scale_factor=4, mode='bilinear'))
            else:
                self.fg_mask_net = Unet(self.n_landmarks, [32, 32, 32, mask_outdim], 3, 3, 32)

        self.sigmoid = nn.Sigmoid()
        self.use_identity_covariance = args['use_identity_covariance']
        #  coordinate grids that we'll need later for estimating gaussians in coordinate space
        with torch.no_grad():
            coord_range = torch.arange(-1, 1, 4.0/self.img_size)
            self.grid_y, self.grid_x = torch.meshgrid(coord_range.float(), coord_range.float())
            self.grid_x = self.grid_x.cuda().contiguous().view(1, 1, -1)
            self.grid_y = self.grid_y.cuda().contiguous().view(1, 1, -1)

            coord_range_2x = torch.arange(-1, 1, 2.0/self.img_size)
            self.grid_y_2x, self.grid_x_2x = torch.meshgrid(coord_range_2x.float(), coord_range_2x.float())
            self.grid_x_2x = self.grid_x_2x.cuda().contiguous().view(1, 1, -1)
            self.grid_y_2x = self.grid_y_2x.cuda().contiguous().view(1, 1, -1)

    def encoding2part_maps(self, encoding, estimate_cov=True):
        '''
        takes in the encoded input B x num_landmarks x H x W , as output from an encoder,
        and creates a B x num_landmarks x H x W normalized landmark heatmaps
        '''
        b, c, h, w = encoding.shape
        if estimate_cov:
            mu_x, mu_y, covar, htmp = estimate_gaussian_params(encoding, self.grid_x, self.grid_y,
                                                               return_covar=True, use_fixed_covar=self.use_identity_covariance,
                                                               fixed_covar=self.fixed_covar)
            return htmp, (mu_x, mu_y, covar)
        else:
            mu_x, mu_y, htmp = estimate_gaussian_params(encoding, self.grid_x, self.grid_y, return_covar=False)
            return htmp, (mu_x, mu_y)

    def construct_downsampling_layers(self):
        """
        parameter-free encoder.
        Just a stack of downsampling layers
        """
        return nn.ModuleList([torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)])

    def pool_appearance_maps(self, featmap, heatmaps):
        """Spatially pools appearance features from featmap based on heatmaps
        Dim-C appearance features
        K target parts/landmarks
        Args:
            featmap (torch.tensor): B x C x H x W feature maps
            heatmaps (torch.tensor): B x K x H x W normalized heatmaps
        Returns:
            torch.tensor: B x K x C pooled appearance vectors
        """

        featmap_expdim = featmap.unsqueeze(1)   # B x 1 x C x H x W
        heatmaps_expdim = heatmaps.unsqueeze(2)  # B x K x 1 x H x W
        fh_prod = featmap_expdim * heatmaps_expdim  # B x K x C x H x W
        return fh_prod.mean(dim=3).mean(dim=3)

    def project_appearance_onto_part_map(self, appearance_vectors, heatmaps):
        """
        Args:
            appearance_vectors (torch.tensor): B x K x C appearance vectors (C-dim vector per part)
            heatmaps (torch.tensor): B x K x H x W normalized heatmaps
        Returns:
            torch.tensor: B x C x H x W projected appearance map (reduced over K)
        """

        # B x K x C x 1 x 1
        appearance_vectors_expdim = appearance_vectors.unsqueeze(3).unsqueeze(3)

        # B x K x 1 x H x W
        heatmaps_expdim = heatmaps.unsqueeze(2)

        # B x K x C x H x W
        ah_prod = appearance_vectors_expdim * heatmaps_expdim
        ah_prod_norm = ah_prod / (1+heatmaps_expdim.sum(1, True))
        return ah_prod_norm.sum(dim=1)  # reduce over k

    def forward(self, color_jittered_input, warped_input=None, cj_gauss_means_x=None, cj_gauss_means_y=None, cj_gauss_covars=None):
        use_input_gaussians = False
        #  terminology: cj (color-jittered, appearance varied), w (warped, pose varied)
        if cj_gauss_means_x is not None:
            #  this block should only happen if we're generating images conditioned on externally provided Gaussians
            #  used as an inference mode
            use_input_gaussians = True
            assert(color_jittered_input is None)
            assert(warped_input is not None)
            assert(cj_gauss_means_y is not None)
            cj_gauss_params = (cj_gauss_means_x, cj_gauss_means_y, cj_gauss_covars)
            cj_part_maps = None

        if not use_input_gaussians:
            color_jittered_input_shape_enc = self.shape_encoder(color_jittered_input)

            color_jittered_input_shape_enc = color_jittered_input_shape_enc[:, 0:self.n_landmarks, :, :]
            cj_part_maps, cj_gauss_params = self.encoding2part_maps(color_jittered_input_shape_enc)

            if warped_input is None:
                return {'vis_centers': (cj_gauss_params[0], cj_gauss_params[1]),
                        'vis_cov': cj_gauss_params[2],
                        'input_a_heatmaps': cj_part_maps}

        warped_input_shape_enc, shape_encoder_first_layer_feats = self.shape_encoder(warped_input, True)
        warped_input_shape_enc = warped_input_shape_enc[:, 0:self.n_landmarks, :, :]
        w_part_maps, w_gauss_params = self.encoding2part_maps(warped_input_shape_enc, True)
        cj_gauss_maps = gaussian_params_to_heatmap(self.grid_x_2x, self.grid_y_2x, cj_gauss_params[0],
                                                   cj_gauss_params[1], cj_gauss_params[2], self.img_size, self.img_size)

        w_gauss_maps = gaussian_params_to_heatmap(self.grid_x_2x, self.grid_y_2x, w_gauss_params[0],
                                                   w_gauss_params[1], w_gauss_params[2], self.img_size, self.img_size)

        #  extract appearance representation from the warped image
        appearance_enc_input = torch.cat((w_part_maps, shape_encoder_first_layer_feats), dim=1)
        appearance_enc = self.appearance_encoder(appearance_enc_input)

        #  spatial average pool over appearance info using original normalized part maps
        #  should be B x K x C
        appearance_vectors = self.pool_appearance_maps(appearance_enc, w_part_maps)

        #  project apearance information onto the heatmap of the color-jittered image
        #  output should be B x C x H x W
        projected_part_map = self.project_appearance_onto_part_map(appearance_vectors, cj_gauss_maps)
        decoded_image = self.image_decoder(cj_gauss_maps, projected_part_map).clone()
        reconstruction = decoded_image
        return_dict = {'reconstruction': reconstruction,
                       'vis_centers': (cj_gauss_params[0], cj_gauss_params[1]),
                       'input_a_gauss_params': cj_gauss_params,
                       'input_a_heatmaps': cj_part_maps,
                       'input_a_gauss_maps': cj_gauss_maps,
                       'input_b_gauss_params': w_gauss_params,
                       'input_b_heatmaps': w_part_maps}
        if self.use_fg_bg_mask:  # if using foreground-background factorization
            foreground_mask = self.sigmoid(self.fg_mask_net(cj_gauss_maps))
            warped_fg_mask = self.sigmoid(self.fg_mask_net(w_gauss_maps))

            background_recon = self.bg_net((1-warped_fg_mask) * warped_input)
            return_dict['reconstruction'] = background_recon * (1-foreground_mask) + decoded_image * foreground_mask
            return_dict['background_recon'] = background_recon
            return_dict['decoded_foreground'] = decoded_image
            return_dict['input_a_fg_mask'] = foreground_mask
            return_dict['input_b_fg_mask'] = warped_fg_mask
        return return_dict
