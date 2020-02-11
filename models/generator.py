"""
Copyright (C) 2019,2020 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Modifications made to adapt the SPADE code to this work.
"""

import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from .normalization import SPADE


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

# Using a stripped down version compared to the version found in
# https://github.com/NVlabs/SPADE/blob/master/models/networks/generator.py
class SPADEGenerator(BaseNetwork):
    def __init__(self, opt, sw, sh):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        # Hard coded here
        self.sw = sw // 16
        self.sh = sh // 16

        # Make the network deterministic by starting with
        # downsampled feature map and segmentation map
        # VAE option removed
        self.fc = nn.Conv2d(self.opt.appearance_nc, 4 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(4 * nf, 4 * nf, opt)

        self.up_0 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_1 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
        final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, seg, appearance):
        """
        Args:
            seg (torch.tensor): segmentation map info. In the case of this work,
                                we use landmark heatmaps (projected gaussians)
            appearance (torch.tensor): appearance information. This tensor holds the appearance
                                       information for each part (ie the style information)
        Returns:
            x (torch.tensor): Rendered output image
        """
        # First downsample the style information
        x_0 = F.interpolate(appearance, size=(self.sh, self.sw))
        x_1 = F.interpolate(appearance, size=(self.sh * 2, self.sw * 2))
        # apply first layers on two scales, 1/16 and 1/8
        # this is a modification made for this work, as the 1/16 scale tends to be too small
        x_0 = self.fc(x_0)
        x_0 = self.head_0(x_0, seg)
        x_1 = self.fc(x_1)
        x_1 = self.head_0(x_1, seg)

        x = 0.5*self.up(x_0) + 0.5*x_1
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))

        # there used to be a tanh here but we're going to make do without it
        return x


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
