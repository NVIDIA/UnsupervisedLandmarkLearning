"""
Custom transformation functions for image augmentation
"""

import random
import numpy as np
from numpy.random import random_sample
import cv2  # for TPS
import torch
import torchvision.transforms as transforms_t
import torchvision.transforms.functional as F


class TPSWarp(object):
    """
    TPS param for non-linear warping:
    nonlinear_pert_range: [-2, 2] (random perturbation of x and y by +/- 2 pixels
    TPS params for affine transformation
    defaults: rotation +/- pi/4
    scales between 0.9 and 1.1 factor
    translates between +/-5 pixels
    """
    def __init__(self, image_size, margin, num_vertical_points, num_horizontal_points,
                 nonlinear_pert_range=[-2, 2],
                 rot_range=[-np.pi/8, np.pi/8],
                 scale_range=[1.05, 1.15],
                 trans_range=[-10, 10], append_offset_channels=False):

        self.nonlinear_pert_range = nonlinear_pert_range
        self.rot_range = rot_range
        self.scale_range = scale_range
        self.trans_range = trans_range
        self.num_points = num_horizontal_points*num_vertical_points
        self.append_offset_channels = append_offset_channels
        horizontal_points = np.linspace(margin, image_size[0] - margin, num_horizontal_points)
        vertical_points = np.linspace(margin, image_size[1] - margin, num_vertical_points)
        xv, yv = np.meshgrid(horizontal_points, vertical_points, indexing='xy')
        xv = xv.reshape(1, -1, 1)
        yv = yv.reshape(1, -1, 1)
        self.grid = np.concatenate((xv, yv), axis=2)
        self.matches = list()

        # TPS define the alignment between source and target grid points
        # here, we just assume nth source keypoint aligns to nth target keypoint
        for i in range(self.num_points):
            self.matches.append(cv2.DMatch(i, i, 0))

    def sample_warp(self):
        """samples the warping matrix based on initialized parameters
        """

        # will be on the right side of the multiply, e.g ([x,y] * w
        rot = random_sample() * (self.rot_range[1] - self.rot_range[0]) + self.rot_range[0]
        sc_x = random_sample() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        sc_y = random_sample() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        t_x = random_sample() * (self.trans_range[1] - self.trans_range[0]) + self.trans_range[0]
        t_y = random_sample() * (self.trans_range[1] - self.trans_range[0]) + self.trans_range[0]
        # return a transposed matrix
        rotscale = [[ sc_x*np.cos(rot), -np.sin(rot)],
               [ np.sin(rot),  sc_y*np.cos(rot)]]
        return rotscale, t_x, t_y

    def random_perturb(self):
        """Returns a matrix for individually perturbing each grid point
        """
        perturb_mat = random_sample(self.grid.shape) * (self.nonlinear_pert_range[1]
                                                        - self.nonlinear_pert_range[0]) + self.nonlinear_pert_range[0]
        return perturb_mat

    def __call__(self, img, tps=None):
        """
        accepts a PIL image
        must convert to numpy array to apply TPS
        converts back to PIL image before returning
        """

        # construct the transformed grid from the regular grid
        img_as_arr = np.transpose(img.numpy(), (1, 2, 0))
        if tps is None:
            warp_matrix, t_x, t_y = self.sample_warp()
            perturb_mat = self.random_perturb()
            center = np.array([[[self.grid[:, :, 0].max()/2.0 + t_x, self.grid[:, :, 1].max()/2.0 + t_y]]])

            target_grid = np.matmul((self.grid - center), warp_matrix) + perturb_mat + center
            tps = cv2.createThinPlateSplineShapeTransformer()
            tps.estimateTransformation(self.grid, target_grid, self.matches)
        img_as_arr = tps.warpImage(img_as_arr, borderMode=cv2.BORDER_REPLICATE)
        dims = img_as_arr.shape

        if self.append_offset_channels:  # extract ground truth warping offsets
            full_grid_x, full_grid_y = np.meshgrid(np.arange(dims[1]), np.arange(dims[0]))
            dims_half_x = dims[1]/2.0
            dims_half_y = dims[0]/2.0
            full_grid_x = (full_grid_x - dims_half_x)/dims_half_x
            full_grid_y = (full_grid_y - dims_half_y)/dims_half_y
            full_grid = np.concatenate((np.expand_dims(full_grid_x, 2), np.expand_dims(full_grid_y, 2)), axis=2)
            img_coord_arr = tps.warpImage(full_grid.astype(np.float32), borderValue=-1024)
            displacement = img_coord_arr
            img_as_arr = np.concatenate((img_as_arr, displacement), 2)

        # convert back to PIL and return
        out_img = torch.from_numpy(img_as_arr).permute(2, 0, 1)
        return out_img


class PairedColorJitter(object):
    """
    Based on the source for torchvision.transforms.ColorJitter
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#ColorJitter
    Modified to apply the same color jitter transformation for a pair of input images
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if value < 0:
            raise ValueError("If {} is a single number, it must be non negative.".format(name))
        value = [center - value, center + value]
        if clip_first_on_zero:
            value[0] = max(value[0], 0)

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img1, img2):

        transforms = []

        brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
        transforms.append(transforms_t.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
        transforms.append(transforms_t.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
        transforms.append(transforms_t.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        hue_factor = random.uniform(self.hue[0], self.hue[1])
        transforms.append(transforms_t.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = transforms_t.Compose(transforms)

        return transform(img1),  transform(img2)
