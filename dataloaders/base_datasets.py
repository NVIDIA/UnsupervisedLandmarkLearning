""" Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Base class for our video dataset
"""

from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from .transforms import TPSWarp, PairedColorJitter


class BaseVideoDataset(Dataset):
    """
    Base dataset class for all video-type datasets in landmark learning
    """
    def __init__(self, args, partition, inference_mode=False):
        super(BaseVideoDataset, self).__init__()
        self.dataset_path = args['dataset_path']
        self.flip_probability = args['flip_probability']
        self.img_size = args['img_size']
        self.inference_mode = inference_mode

        self.num_frames_array = self.setup_frame_array(args, partition)
        assert(self.num_frames_array[0] == 0)

        # video frame folders
        self.resize = transforms.Resize([self.img_size, self.img_size])
        self.to_tensor = transforms.ToTensor()
        self.paired_color_jitter = PairedColorJitter(0.4, 0.4, 0.4, 0.3)
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.3)
        self.TPSWarp = TPSWarp([self.img_size, self.img_size], 10, 10, 10,
                               rot_range=[args['rot_lb'], args['rot_ub']],
                               trans_range=[args['trans_lb'], args['trans_ub']],
                               scale_range=[args['scale_lb'], args['scale_ub']],
                               append_offset_channels=True)

        self.normalize = transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5))

    def setup_frame_array(self, args):
        """
        Implement this function to setup the cummulative array
        cummulative array should have N+1 bins, where N is the number of videos
        first bin should be 0
        last bin should be the total number of frames in the dataset
        Also use this function to setup any dataset-specific fields
        """
        pass

    def __len__(self):
        """
        returns length of dataset (total number of frames)
        """
        return self.num_frames_array[-1]

    def get_frame_index(self, global_idx):
        """maps global frame index to video index and local video frame index
        """
        vid_idx = np.searchsorted(self.num_frames_array, global_idx, side='right')-1
        frame_idx = global_idx - self.num_frames_array[vid_idx]
        return vid_idx, frame_idx

    def process_batch(self, vid_idx, img_idx):
        """
        implement this function
        extracts the requisite frames from the dataset
        returns a dictionary that must include entries in the required_keys variable in __getitem__
        """
        pass

    def sample_temporal(self, num_vid_frames, img_idx, range_min, range_max):
        """
        samples another frame from the same video sequence
        num_vid_frames: num frames in current video segment
        range_min: minimum sampling offset (must be at least this far away)
        range_max: maximum sampling offset
        """
        if num_vid_frames - img_idx > range_min:
            idx_offset = np.random.randint(range_min, min(range_max, num_vid_frames-img_idx))
        else:
            # sample in the opposite direction
            idx_offset = -min(img_idx, np.random.randint(range_min, range_max))
        return idx_offset

    def construct_color_warp_pair(self, img):
        """
        given an input image
        constructs the color jitter - warping training pairs
        returns color jittered, warped image, warping flow, and target tensors
        """
        img_color_jittered = self.to_tensor(self.resize(self.color_jitter(img)))
        img = self.to_tensor(self.resize(img))
        img_warped = self.TPSWarp(img)
        img_warped_offsets = img_warped[3:]
        img_warped = self.normalize(img_warped[0:3])
        img_color_jittered = self.normalize(img_color_jittered)
        target = self.normalize(img)
        return img_color_jittered, img_warped, img_warped_offsets, target

    def __getitem__(self, idx):

        # convert global index to video and local frame index
        vid_idx, img_idx = self.get_frame_index(idx)
        out_dict = self.process_batch(vid_idx, img_idx)
        # assert all required keys are present
        # construct the batch
        if self.inference_mode:
            required_keys = {'input_a', 'vid_idx', 'img_idx'}
        else:
            required_keys = {'input_a', 'input_b', 'target', 'input_temporal', 'imname'}
        assert(len(required_keys - out_dict.keys()) == 0)

        return out_dict
