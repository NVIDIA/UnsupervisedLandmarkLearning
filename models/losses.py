"""Implementation for various loss modules
GAN loss adapted from pix2pixHD (see comment below)
"""

import torch
import torch.nn as nn

from torch.autograd import Variable
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.MSELoss = torch.nn.MSELoss()

    def normalize_batch(self, batch, div_factor=255.):
        # normalize using imagenet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch = torch.div(batch, div_factor)
        batch -= mean
        batch = torch.div(batch, std)
        return batch

    def forward(self, x, y):
        x = self.normalize_batch(x)
        y = self.normalize_batch(y)
        return self.L1Loss(x, y)


# for reference
indx2name = {0: 'conv1_1', 1: 'relu1_1', 2: 'conv1_2', 3: 'relu1_2',
             4: 'pool1', 5: 'conv2_1', 6: 'relu2_1', 7: 'conv2_2', 8: 'relu2_2', 9: 'pool2',
             10: 'conv3_1', 11: 'relu3_1', 12: 'conv3_2', 13: 'relu3_2', 14: 'conv3_3',
             15: 'relu3_3', 16: 'conv3_4', 17: 'relu3_4', 18: 'pool3',
             19: 'conv4_1', 20: 'relu4_1', 21: 'conv4_2', 22: 'relu4_2', 23: 'conv4_3',
             24: 'relu4_3', 25: 'conv4_4', 26: 'relu4_4', 27: 'pool4',
             28: 'conv5_1', 29: 'relu5_1', 30: 'conv5_2', 31: 'relu5_2', 32: 'conv5_3',
             33: 'relu5_3', 34: 'conv5_4', 35: 'relu5_4'}


# keep 3, 8, 13, 22
class Vgg19PerceptualLoss(PerceptualLoss):
    def __init__(self, reduced_w, layer_name='relu5_2'):
        super(Vgg19PerceptualLoss, self).__init__()
        self.vgg19_layers = nn.Sequential(*list(models.vgg19(pretrained=True).features.children())[:23])
        self.MSELoss = torch.nn.MSELoss()
        # set hooks on layers indexed 3, 8, 13 and 22
        # registers the hook to target layers
        # allowing us to extract the outputs and store in
        # self.extracted_feats
        # be sure to clear self.extracted_feats
        # before use
        self.extracted_feats = []

        def feature_extract_hook(module, inputs, outputs):
            self.extracted_feats.append(outputs)
        self.extract_layers = [3, 8, 13, 22]  # assume last one will be input layer 0
        if reduced_w:
            self.loss_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        else:
            self.loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        for layer_id in self.extract_layers:
            self.vgg19_layers[layer_id].register_forward_hook(feature_extract_hook)

        # disable grad for all VGG params
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y, div_factor=1):
        x[y == 0] = 0.0

        self.extracted_feats = []
        _ = self.vgg19_layers(x)
        x_feats = self.extracted_feats
        x_feats.append(x)
        self.extracted_feats = []
        _ = self.vgg19_layers(y)
        y_feats = self.extracted_feats
        y_feats.append(y)

        layer_mse_losses = []
        for i in range(len(x_feats)):
            layer_mse_losses.append(self.MSELoss(x_feats[i], y_feats[i]))

        full_loss = 0.0
        for i in range(len(x_feats)):
            full_loss += self.loss_weights[i] * layer_mse_losses[i]

        return full_loss


"""
Adapted from
https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
"""
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real).cuda()
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real).cuda()
            return self.loss(input[-1], target_tensor)
