"""
Utility functions for visualization and image dumping
"""

from utils.utils import denormalize_batch
from PIL import Image
from PIL import ImageDraw
import numpy as np


def uint82bin(n, count=8):
    """adapted from https://github.com/ycszen/pytorch-segmentation/blob/master/transform.py
    returns the binary of integer n, count refers to amount of bits"""

    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def generate_palette():
    """adapted from https://github.com/ycszen/pytorch-segmentation/blob/master/transform.py
    Used to generate the color palette we use to plot colorized heatmaps
    """

    N = 41
    palettes = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i+1
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        palettes[i, 0] = r
        palettes[i, 1] = g
        palettes[i, 2] = b
    palettes = palettes.astype(np.float)
    return palettes


def dump_image(normalized_image, landmark_coords, out_name=None):
    """Denormalizes the output image and optionally plots the landmark coordinates onto the image

    Args:
        normalized_image (torch.tensor): Image reconstruction output from the model (normalized)
        landmark_coords (torch.tensor): x, y coordinates in normalized range -1 to 1
        out_name (str, optional): file to write to
    Returns:
        np.array: uint8 image data stored in numpy format
    """
    warped_image = np.clip(denormalize_batch(normalized_image).data.cpu().numpy(), 0, 1)
    img = Image.fromarray((warped_image.transpose(1, 2, 0)*255).astype(np.uint8))

    if landmark_coords is not None:
        xs = landmark_coords[0].data.cpu().numpy()
        ys = landmark_coords[1].data.cpu().numpy()
        h, w = img.size
        draw = ImageDraw.Draw(img)
        for i in range(len(xs)):
            x_coord = (xs[i] + 1) * h // 2
            y_coord = (ys[i] + 1) * w // 2
            draw.text((x_coord, y_coord), str(i), fill=(0, 0, 0, 255))
    if out_name is not None:
        img.save(out_name)
    return np.array(img)


def project_heatmaps_colorized(heat_maps):
    color_palette = generate_palette()
    c, h, w = heat_maps.shape
    heat_maps = heat_maps / heat_maps.max()
    heat_maps_colored = np.matmul(heat_maps.reshape(c, h*w).transpose(1, 0), color_palette[0:c, :])
    heat_maps_out = np.clip(heat_maps_colored.transpose(1, 0).reshape(3, h, w), 0, 255)
    return heat_maps_out
