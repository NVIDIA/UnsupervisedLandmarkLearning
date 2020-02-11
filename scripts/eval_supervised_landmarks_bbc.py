"""
This script should exactly replicate the matlab-based evaluation provided in the BBCPose
eval toolkit, thresholded at a 6 pixel radius
"""
import scipy.io as sio
import numpy as np
import argparse
import os


def eval_kpts(my_preds, anno_gt):
    preds = 0.5 * (my_preds['my_pred']-1) + 1
    gt = 0.5 * (anno_gt-1) + 1
    err = np.sqrt(np.sum((preds - gt) ** 2, axis=0))
    #  error if regressed position is greater than 6 pixels away from ground truth
    err[err <= 6] = 1
    err[err > 6] = 0
    return err


def main(results_path, test_anno_path= None, val_anno_path=None):
    if test_anno_path is not None:  # if test annotations are provided
        test_pred_results = os.path.join(results_path, 'preds_test.mat')
        test_gt = sio.loadmat(test_anno_path)
        test_preds = sio.loadmat(test_pred_results)
        err = eval_kpts(test_preds, test_gt['results']['gt'][0, 0])
        print('Test', err.mean())
    if val_anno_path is not None:  # if validation set annotations are provided
        val_pred_results = os.path.join(results_path, 'preds_val.mat')
        val_gt = sio.loadmat(val_anno_path)
        val_preds = sio.loadmat(val_pred_results)
        err = eval_kpts(val_preds, val_gt['my_pred'])
        print('Val', err.mean())


if __name__ == '__main__':
    #  load yaml
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--results_path', type=str, help="Path to where the regressed landmark results are stored")
    parser.add_argument('--test_anno_path', type=str, default=None, help="Location of the results.mat file provided by the bbcpos evaluation toolkit")
    parser.add_argument('--val_anno_path', type=str, default=None, help="Location of the validation set annotations stored in the same format as results.mat")
    args = parser.parse_args()
    main(args.results_path, args.test_anno_path, args.val_anno_path)
