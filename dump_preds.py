""" Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

This script is for dumping the gaussian landmark predictions,
provided the dataset and trained landmark model checkpoint.
This is necessary for evaluating the landmark quality (on BBC)
as well as for performing the video manipulation tasks.
"""
import torch
import argparse
import pickle
import os
import numpy as np
from torch.utils.data import DataLoader
from dataloaders.bbc_pose_dataset import BBCPoseLandmarkEvalDataset
from utils.utils import parse_all_args, load_weights, get_model
from copy import deepcopy


def setup_dataloaders(config):
    #  setup the dataset
    num_workers = config['num_workers']
    val_dataloader = None
    # For both video manipulation and landmark evaluation (regression to annotated keypoints)
    if config['dataset'] == 'bbc_pose':
        train_dataset = BBCPoseLandmarkEvalDataset(config, 'train')
        val_dataset = BBCPoseLandmarkEvalDataset(config, 'val')
        test_dataset = BBCPoseLandmarkEvalDataset(config, 'test')
        # validation set for model selection based on landmark evaluation.
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                    shuffle=False, num_workers=num_workers)
    else:
        print("unrecognized dataset!")
        exit(1)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'],
                                 shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader, val_dataloader


def convert_encoding(config, model, dataloader):
    """
    iterate the data and extract the tensors we need
    """
    all_preds = []
    all_preds_cov = []
    all_preds_cov_values = []
    all_vid_idx = []
    all_frame_idx = []
    all_gt_kpts = []
    all_bboxes = []

    print('length of dataset: ', len(dataloader))
    for cnt, curr_batch in enumerate(dataloader):
        if cnt % 10 == 0:
            print('cnt', cnt, 'total', len(dataloader))

        # im will be b x c x 128 x 128
        # gt_keypoints will be b x 10
        # this avoids a shared memory problem when num_workers > 0 (hopefully)
        curr_batch_cpy = deepcopy(curr_batch)
        del curr_batch
        curr_batch = curr_batch_cpy
        vid_idx = deepcopy(curr_batch['vid_idx']).numpy()
        frame_idx = curr_batch['img_idx'].numpy()
        im = deepcopy(curr_batch['input_a'])

        if config['dataset'] == 'bbc_pose':
            all_gt_kpts.append(curr_batch['gt_kpts'].numpy())
            all_bboxes.append(curr_batch['bbox'].numpy())

        output_dict = model(im.cuda())
        heatmap_centers = output_dict['vis_centers']
        heatmap_centers_x = heatmap_centers[0].cpu()
        heatmap_centers_y = heatmap_centers[1].cpu()
        heatmap_cov = output_dict['vis_cov'].cpu()

        heatmap_centers_cat = torch.cat((heatmap_centers_x, heatmap_centers_y), 1)
        all_vid_idx.append(vid_idx)
        all_frame_idx.append(frame_idx)
        all_preds.append(heatmap_centers_cat.cpu().detach().numpy().astype('float16'))
        # if cov is fitted, save original and after decomposing
        if not config['use_identity_covariance']:
            cov_chol = torch.cholesky(heatmap_cov)
            all_preds_cov_values.append(cov_chol.cpu().detach().numpy().astype('float16'))
            all_preds_cov.append(heatmap_cov.detach().numpy().astype('float16'))

    all_preds_cat = np.concatenate(all_preds, 0)
    all_vid_idx = np.concatenate(all_vid_idx, 0)
    all_frame_idx = np.concatenate(all_frame_idx, 0)

    # currently only bbc has GT keypoints for evaluation
    if config['dataset'] == 'bbc_pose':
        all_bboxes = np.concatenate(all_bboxes, 0)
        all_gt_kpts = np.concatenate(all_gt_kpts, 0)
    if not config['use_identity_covariance']:
        all_preds_cov_values = np.concatenate(all_preds_cov_values, 0)
        all_preds_cov = np.concatenate(all_preds_cov, 0)
    return all_preds_cat, all_preds_cov, all_preds_cov_values, all_vid_idx, all_frame_idx, all_bboxes, all_gt_kpts


def save_files(x, x_cov, x_cov_values, vid, frame, bboxes, gt, out_dir):
    results = {}
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    outname = os.path.join(out_dir, 'gaussians.pkl3')
    results['predictions_mean'] = x
    results['predictions_cov'] = x_cov
    results['predictions_cov_decomp'] = x_cov_values
    results['vid'] = vid
    results['frame'] = frame
    results['bboxes'] = bboxes
    results['gt'] = gt

    with open(outname, 'wb') as handle:
        pickle.dump(results, handle, protocol=3)


def eval_encoding(config, model, train_dataloader, test_dataloader, val_dataloader):
    preds_cat, preds_cov, preds_cov_values, vid_idx, frame_idx, bboxes, gt = convert_encoding(config, model, test_dataloader)
    #  multiply test ground truth keypoints by 0 to avoid any potential leakage of test annotations
    save_files(preds_cat, preds_cov, preds_cov_values, vid_idx, frame_idx, bboxes, 0*gt, config['gaussians_save_path'] + '/test')

    preds_cat, preds_cov, preds_cov_values, vid_idx, frame_idx, bboxes, gt = convert_encoding(config, model, train_dataloader)
    save_files(preds_cat, preds_cov, preds_cov_values, vid_idx, frame_idx, bboxes, gt, config['gaussians_save_path'] + '/train')

    if val_dataloader is not None:
        # multiply val ground truth keypoints by 0 to avoid any potential leakage of validation set annotations
        preds_cat, preds_cov, preds_cov_values, vid_idx, frame_idx, bboxes, gt = convert_encoding(config, model, val_dataloader)
        save_files(preds_cat, preds_cov, preds_cov_values, vid_idx, frame_idx, bboxes, 0*gt, config['gaussians_save_path'] + '/val')


def main(config):
    print(config)
    # initialize model
    model = get_model(config)
    # load weights from checkpoint
    state_dict = load_weights(config['resume_ckpt'])
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    train_dataloader, test_dataloader, val_dataloader = setup_dataloaders(config)
    eval_encoding(config, model, train_dataloader, test_dataloader, val_dataloader)


if __name__ == '__main__':
    # load yaml
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str)
    parser.add_argument('--gaussians_save_path', type=str)
    config, args = parse_all_args(parser, 'configs/defaults.yaml', return_args=True)
    config['gaussians_save_path'] = args.gaussians_save_path
    config['no_verbose'] = True
    main(config)
