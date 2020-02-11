""" Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

This file contains subroutines for our training pipeline
"""

import torch
import torch.nn as nn


def conv_ReLU(in_channels, out_channels, kernel_size, stride=1, padding=0,
              use_norm=True, norm=nn.InstanceNorm2d):
    """Returns a 2D Conv followed by a ReLU
    """
    if use_norm:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding),
                             norm(out_channels),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding),
                             nn.ReLU(inplace=True))


def estimate_gaussian_params(in_heatmaps, grid_x, grid_y, return_covar=False, skip_norm=False, activation=torch.exp, use_fixed_covar=False, fixed_covar=0.1):
    """Converts heatmaps to 2D Gaussians by estimating mean and covariance

    Args:
        in_heatmaps: b x K x H x W heatmaps
        grid_x, grid_y: 1 x 1 x (HW) We compute mean and covariance over these
        return_covar (bool): if true, also return the covariance matrix
        activation: activation function for logits. Defaults to torch.exp, which gives us a softmax
        use_fixed_covar (bool): if true, return hard coded scaled identity matrix, otherwise estimate it from the heatmap
    """
    b, c, h, w = in_heatmaps.shape
    heatmaps_reshaped = in_heatmaps.view(b, c, -1)
    # should be b x c x HW
    if skip_norm:
        heatmaps_norm = heatmaps_reshaped
    else:
        heatmaps_norm = activation(heatmaps_reshaped)
        heatmaps_norm = heatmaps_norm / heatmaps_norm.sum(2, True)

    mu_x = torch.sum(heatmaps_norm * grid_x, 2)
    mu_y = torch.sum(heatmaps_norm * grid_y, 2)

    if return_covar:
        if use_fixed_covar:  # generate a fixed diagonal covariance matrix
            covar = torch.eye(2, 2, device=torch.cuda.current_device()).view(1, 1, 2, 2) * fixed_covar
        else:  # actually estimate the covariance from the heatmaps
            # should be 1 x 1 x 2 x HW
            coord_grids_xy = torch.cat((grid_x, grid_y), dim=1).unsqueeze(0)
            # covar will be b x 1 x 2 x 2
            mu = torch.stack((mu_x, mu_y), 2).view(b, c, 2, 1)
            mu_outer = torch.matmul(mu, torch.transpose(mu, 2, 3))
            covar = torch.matmul(coord_grids_xy * heatmaps_norm.unsqueeze(2), coord_grids_xy.transpose(2, 3)) - mu_outer
        return mu_x, mu_y, covar, heatmaps_norm.view(b, c, h, w)

    return mu_x, mu_y, heatmaps_norm.view(b, c, h, w)


def gaussian_params_to_heatmap(grid_x, grid_y, mu_x, mu_y, covar, out_h, out_w):
    """Converts Gaussian parameters to heatmaps

    Args:
        grid_x, grid_y: 1 x 1 x (HW)
        mu_x, mu_y: B x K
        covar: B x K x 2 x 2
    """
    # B x K x HW
    B, K = mu_x.shape
    xx = grid_x - mu_x.unsqueeze(2)
    yy = grid_y - mu_y.unsqueeze(2)
    # B x K x HW x 2
    xxyy_t = torch.stack((xx, yy), dim=3)
    covar_inv = torch.inverse(covar)
    new_dist = xxyy_t*torch.matmul(xxyy_t, covar_inv)

    new_dist_norm = 1.0/(1+new_dist.sum(3))
    new_dist_rshp = new_dist_norm.view(B, K, out_h, out_w)
    return new_dist_rshp


class MyUpsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.upsample = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = x.float()
        if self.mode == 'bilinear':
            x = self.upsample(x, scale_factor=self.scale_factor, mode=self.mode,
                              align_corners=True)
        else:
            x = self.upsample(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


def decoder_block(in_filters, out_filters, transpose=False, norm=nn.InstanceNorm2d):
    if transpose:
        return nn.Sequential(nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
                             norm(out_filters),
                             nn.ReLU(inplace=True))

    else:
        return nn.Sequential(conv_ReLU(in_filters, out_filters, 3, stride=1, padding=1, use_norm=True, norm=norm),
                             MyUpsample(scale_factor=2, mode='bilinear'),
                             conv_ReLU(out_filters, out_filters, 3, stride=1, padding=1, use_norm=True, norm=norm))


def encoder_block(in_filters, out_filters, norm=nn.InstanceNorm2d):
    """helper function to return two 3x3 convs with the 1st being stride 2
    """
    return nn.Sequential(conv_ReLU(in_filters, out_filters, 3, stride=2, padding=1, use_norm=True, norm=norm),
                         conv_ReLU(out_filters, out_filters, 3, stride=1, padding=1, use_norm=True, norm=norm))
