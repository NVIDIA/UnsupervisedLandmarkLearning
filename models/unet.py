"""Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Parameterized Unet module that we use to construct our shape and appearance encoders
"""

import torch.nn as nn
from .submodules import conv_ReLU,  encoder_block, decoder_block


class Unet(nn.Module):
    def __init__(self, num_input_channels, decoder_out_channels, num_downsamples, num_upsamples, filters):
        super(Unet, self).__init__()
        # decoder will have 1 fewer upsamples than the encoder if True
        self.num_downsamples = num_downsamples
        self.num_upsamples = num_upsamples
        self.decoder_out_channels = decoder_out_channels
        self.norm = nn.InstanceNorm2d
        # encoder will just be a set of downsampling layers if False
        assert(len(decoder_out_channels) == num_upsamples + 1)
        self.encoder_layers, per_layer_channels, f_block_filters = self.construct_encoder(num_input_channels, filters)

        self.decoder_layers, self.skip_convs = self.construct_decoder(per_layer_channels, self.decoder_out_channels)
        assert(len(self.skip_convs) == num_upsamples)
        self.f_filters = f_block_filters

    def construct_encoder(self, num_input_channels, filters):
        """Helper function to return the encoder layers
        Args:
            num_input_channels (int): number of inuput channels to the encoder.
            filters (int): 1st encoder layer feature dimension (doubles every layer).

        Returns:
            torch.nn.ModuleList: module list of encoder layers
            per_channel_filters (List(int)): List to keep track of feature dimensions per layer.
            f_block_filters (int): feature dimension output of the first convolutional block
        """
        if filters == 0:
            filters = num_input_channels * 2
        conv_1 = conv_ReLU(num_input_channels, filters, 3, stride=1, padding=1, norm=self.norm)
        conv_2 = conv_ReLU(filters, filters*2, 3, stride=1, padding=1, norm=self.norm)
        conv_2B = encoder_block(filters*2, filters*4, norm=self.norm)
        layer_list = []
        per_channel_filters = []
        layer_list.append(nn.Sequential(conv_1, conv_2, conv_2B))
        per_channel_filters.append(filters*4)
        filters = filters * 4
        f_block_filters = filters
        for ds in range(self.num_downsamples-1):
            layer_list.append(encoder_block(filters, filters*2, norm=self.norm))
            filters = filters * 2
            per_channel_filters.append(filters)
        # return as a list such that we may need to index later for skip layers
        return nn.ModuleList(layer_list), per_channel_filters, f_block_filters

    def construct_decoder(self, enc_plc, decoder_out_channels):
        """
        helper function to return upsampling convs for the decoder
        enc_plc: encoder per-layer channels
        output_channels: number of channels to output at final layer
        """

        output_list = []
        skip_convs = []
        enc_plc_rev = enc_plc[::-1]

        # first take in last output from encoder
        in_channels = enc_plc_rev[0]

        for us in range(self.num_upsamples+1):
            if us == 0: # first one just conv
                output_list.append(conv_ReLU(in_channels, in_channels, 1))
            else:
                out_channels = decoder_out_channels[us-1]
                mapping_conv = conv_ReLU(enc_plc_rev[us-1], in_channels, 1, use_norm=False)
                # map encoder outputs to match current inputs
                if us == self.num_upsamples: # if last one
                    dec_layer = nn.Sequential(decoder_block(in_channels, out_channels),
                                              nn.Conv2d(out_channels, decoder_out_channels[-1], 1))
                else:
                    dec_layer = decoder_block(in_channels, out_channels)
                output_list.append(dec_layer)
                skip_convs.append(mapping_conv)

                in_channels = out_channels

        return nn.ModuleList(output_list), nn.ModuleList(skip_convs)

    def forward(self, input, output_first_featmap=False):
        encoder_outputs = []
        output = input

        # encode, and save the per-layer outputs
        for layer in self.encoder_layers:
            output = layer(output)
            encoder_outputs.append(output)

        for i in range(len(self.decoder_layers)):
            if i == 0:
                output = self.decoder_layers[i](encoder_outputs[-1])
            else:
                # apply skip conv on input
                encoder_skip_feats = self.skip_convs[i-1](encoder_outputs[-i])
                output = self.decoder_layers[i](output + encoder_skip_feats)
        if output_first_featmap:
            return output, encoder_outputs[0]

        return output
