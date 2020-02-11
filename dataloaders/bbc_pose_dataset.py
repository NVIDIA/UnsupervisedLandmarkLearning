"""Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Dataset classes for handling the BBCPose data
"""

from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import scipy.io as sio
from .base_datasets import BaseVideoDataset


class BBCPoseDataset(BaseVideoDataset):
    def __init__(self, args, partition):
        super(BBCPoseDataset, self).__init__(args, partition)

    def setup_frame_array(self, args, partition):
        if partition == 'train':
            self.input_vids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        elif partition == 'validation':
            self.input_vids = ['11', '12', '13', '14', '15']

        # load the annotations file
        self.annos = sio.loadmat(os.path.join(self.dataset_path, 'code', 'bbcpose.mat'))['bbcpose'][0]
        # first bin is 0
        self.num_frames_array = [0]
        frac = 1
        if partition == 'val':
            frac = args['val_frac']
        for folder in self.input_vids:
            curr_vid_anno = self.annos[int(folder)-1]
            # truncate validation if frac is specified
            self.num_frames_array.append(int(curr_vid_anno[3].shape[1]*frac))
        self.num_frames_array = np.array(self.num_frames_array).cumsum()

        return self.num_frames_array

    def process_batch(self, vid_idx, img_idx):
        vid_path = os.path.join(self.dataset_path, self.input_vids[vid_idx])
        curr_vid_anno = self.annos[int(self.input_vids[vid_idx])-1]
        num_frames = curr_vid_anno[3].shape[1]

        img_idx2_offset = self.sample_temporal(num_frames, img_idx, 3, 40)
        gt_kpts = curr_vid_anno[4][:, :, img_idx].copy()

        img_1 = os.path.join(vid_path, str(int(curr_vid_anno[3][0][img_idx])) + '.jpg')
        img_2 = os.path.join(vid_path, str(int(curr_vid_anno[3][0][img_idx + img_idx2_offset])) + '.jpg')

        bbox_x_min = gt_kpts[0].min() - 60
        bbox_x_max = gt_kpts[0].max() + 60
        bbox_y_min = gt_kpts[1].min() - 60
        bbox_y_max = gt_kpts[1].max() + 60

        # clip the bounding boxes
        img_a = Image.open(img_1).convert('RGB')
        wh = img_a.size
        bbox_x_min = max(0, bbox_x_min)
        bbox_y_min = max(0, bbox_y_min)
        bbox_x_max = min(wh[0], bbox_x_max)
        bbox_y_max = min(wh[1], bbox_y_max)

        bbox_a = (bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max)
        img_a = img_a.crop(bbox_a)
        img_temporal = Image.open(img_2).convert('RGB')
        img_temporal = img_temporal.crop(bbox_a)
        # randomly flip
        if np.random.rand() <= self.flip_probability:
            # flip both images
            img_a = transforms.functional.hflip(img_a)
            img_temporal = transforms.functional.hflip(img_temporal)

        bbox_w = bbox_x_max - bbox_x_min
        bbox_h = bbox_y_max - bbox_y_min

        img_temporal = self.to_tensor(self.resize(img_temporal))
        img_temporal = self.normalize(img_temporal)

        img_a_color_jittered, img_a_warped, img_a_warped_offsets, target=self.construct_color_warp_pair(img_a)

        # 2x7 array of x and y
        gt_kpts_normalized = gt_kpts.copy()
        gt_kpts_normalized[0] = ((gt_kpts_normalized[0] - bbox_x_min) / bbox_w - 0.5) * 2
        gt_kpts_normalized[1] = ((gt_kpts_normalized[1] - bbox_y_min) / bbox_h - 0.5) * 2
        return {'input_a': img_a_color_jittered, 'input_b': img_a_warped,
                'input_temporal': img_temporal, 'target': target,
                'imname': self.input_vids[vid_idx] + '_' + str(img_idx) + '.jpg',
                'warping_tps_params': img_a_warped_offsets, 'gt_kpts': gt_kpts_normalized}


class BBCPoseLandmarkEvalDataset(Dataset):
    def __init__(self, args, partition='train'):
        super(BBCPoseLandmarkEvalDataset, self).__init__()

        self.partition = partition
        self.dataset_path = args['dataset_path']
        self.img_size = args['img_size']

        # if test set, load from the .mat annotation file
        if self.partition == 'test':
            self.annos = sio.loadmat(os.path.join(self.dataset_path, 'code', 'results.mat'))['results'][0][0]
            self.gt = self.annos[0]
            self.frame_name = self.annos[1][0]
            self.video_name = self.annos[2][0]
            self.img_len = len(self.video_name)
        else:
            self.annos = sio.loadmat(os.path.join(self.dataset_path, 'code', 'bbcpose.mat'))['bbcpose'][0]
            self.num_frames_array = [0]
            if partition == 'train_heldout':
                # some frames from here are unannotated and therefore unused during training
                # useful for testing performance on novel frames of the training subjects
                self.input_vids = ['4', '6', '8', '10']
                for folder in self.input_vids:
                    self.num_frames_array.append(20)
            else:
                if partition == 'train':
                    self.input_vids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
                elif partition == 'val':
                    self.input_vids = ['11', '12', '13', '14', '15']

                for folder in self.input_vids:
                    curr_vid_anno = self.annos[int(folder)-1]
                    self.num_frames_array.append(curr_vid_anno[3].shape[1])
            self.num_frames_array = np.array(self.num_frames_array).cumsum()
            self.img_len = self.num_frames_array[-1]

        self.orig_transforms = transforms.Compose([
            transforms.Resize([self.img_size, self.img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

    def __len__(self):
        return self.img_len

    def get_frame_index(self, global_idx):
        vid_idx = np.searchsorted(self.num_frames_array, global_idx, side='right')-1
        frame_idx = global_idx - self.num_frames_array[vid_idx]
        return vid_idx, frame_idx

    def __getitem__(self, idx):
        if self.partition == 'test':
            vid_idx, img_idx = self.video_name[idx], self.frame_name[idx]
            img_1 = os.path.join(self.dataset_path, str(vid_idx), str(img_idx) + '.jpg')
            gt_kpts = self.gt[:, :, idx]
        elif self.partition == 'train_heldout':
            vid_idx, img_idx = self.get_frame_index(idx)
            print('vid_idx', vid_idx, 'img_idx', img_idx)
            vid_path = os.path.join(self.dataset_path, self.input_vids[vid_idx])
            curr_vid_anno = self.annos[int(self.input_vids[vid_idx])-1]
            gt_kpts = curr_vid_anno[4][:, :, 0] # take the first box
            img_1 = os.path.join(vid_path, str(img_idx+1) + '.jpg')
        else:
            vid_idx, img_idx = self.get_frame_index(idx)
            vid_path = os.path.join(self.dataset_path, self.input_vids[vid_idx])
            curr_vid_anno = self.annos[int(self.input_vids[vid_idx])-1]
            gt_kpts = curr_vid_anno[4][:, :, img_idx]
            img_1 = os.path.join(vid_path, str(int(curr_vid_anno[3][0][img_idx])) + '.jpg')

        with Image.open(img_1) as img:
            img = img.convert('RGB')
            wh = img.size  # width, height of image
            box_x_1 = gt_kpts[0].min()
            box_x_2 = gt_kpts[0].max()
            box_y_1 = gt_kpts[1].min()
            box_y_2 = gt_kpts[1].max()

            box_x_center = (box_x_1 + box_x_2)/2
            box_y_center = (box_y_1 + box_y_2)/2

            bbox_x_min = max(0, box_x_center - 150)
            bbox_x_max = min(wh[0], box_x_center + 150)
            bbox_y_min = max(0, box_y_center - 150)
            bbox_y_max = min(wh[1], box_y_center + 150)

            target_keypoints = gt_kpts.copy()

            bbox = (bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max)
            img = img.crop(bbox)
            
            bbox_w = bbox_x_max - bbox_x_min
            bbox_h = bbox_y_max - bbox_y_min
            # center coordinate space
            target_keypoints[0] = (target_keypoints[0] - bbox_x_min - bbox_w/2) / bbox_w
            target_keypoints[1] = (target_keypoints[1] - bbox_y_min - bbox_h/2) / bbox_h
            target_keypoints = torch.flatten(torch.FloatTensor(target_keypoints))
            img = self.orig_transforms(img)
            return {'input_a': img, 'gt_kpts': target_keypoints, 'vid_idx': vid_idx, 'img_idx': img_idx, 'bbox': np.array(bbox)}
