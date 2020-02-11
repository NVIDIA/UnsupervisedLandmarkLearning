"""Script for fitting the regression paramaters mapping the landmarks to the
annotated keypoints on BBC. Regression results are saved in a .mat file.
Run eval_supervised_landmarks_bbc.py to evaluate the regressed keypoints.
"""
import numpy as np
import sklearn.linear_model
import argparse
import scipy.io as sio
import pickle
import os


def save_results(y_predict, all_bboxes, output_path):
    box_w = all_bboxes[:, 2] - all_bboxes[:, 0]
    box_h = all_bboxes[:, 3] - all_bboxes[:, 1]
    box_w = np.expand_dims(np.expand_dims(box_w, axis=1), axis=2)
    box_h = np.expand_dims(np.expand_dims(box_h, axis=1), axis=2)

    # B x 2 X 1
    box_wh = np.concatenate((box_w, box_h), 1)

    box_x_min = all_bboxes[:, 0]
    box_y_min = all_bboxes[:, 1]
    box_x_min = np.expand_dims(np.expand_dims(box_x_min, axis=1), axis=2)
    box_y_min = np.expand_dims(np.expand_dims(box_y_min, axis=1), axis=2)
    box_mins = np.concatenate((box_x_min, box_y_min), 1)

    y_predict = y_predict * box_wh + box_wh/2
    y_predict += box_mins

    y_predict = np.transpose(y_predict, (1, 2, 0))

    predictions = {}
    predictions['my_pred'] = y_predict
    print(output_path)
    sio.savemat(output_path + '.mat', predictions)


def load_encoding(path):
    print(path)
    with open(path, 'rb') as handle:
        files = pickle.load(handle)
    return files


def main(gaussian_path, output_path):
    training_data = load_encoding(os.path.join(gaussian_path, 'train', 'gaussians.pkl3'))
    testing_data = load_encoding(os.path.join(gaussian_path, 'test', 'gaussians.pkl3'))
    val_data = load_encoding(os.path.join(gaussian_path, 'val', 'gaussians.pkl3'))

    X_train, Y_train = training_data['predictions_mean'], training_data['gt']
    X_val = val_data['predictions_mean']
    X_test = testing_data['predictions_mean']

    #  Following the same procedure as https://github.com/tomasjakab/imm/blob/dev/scripts/test.py
    #  from Tomas Jakab
    regr = sklearn.linear_model.Ridge(alpha=0.0, fit_intercept=False)
    print(X_train.shape, Y_train.shape)
    print("Fitting...")
    _ = regr.fit(X_train, Y_train)
    print("Predicting on test...")
    y_predict_test = regr.predict(X_test)

    print("Predicting on Validation...")
    y_predict_val = regr.predict(X_val)

    n_keypoints = 7
    y_predict_test_rshp = y_predict_test.reshape(-1, 2, n_keypoints)
    y_predict_val_rshp = y_predict_val.reshape(-1, 2, n_keypoints)

    save_results(y_predict_test_rshp, testing_data['bboxes'], os.path.join(output_path, 'preds_test'))
    save_results(y_predict_val_rshp, val_data['bboxes'], os.path.join(output_path, 'preds_val'))


if __name__ == '__main__':
    # load yaml
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gaussian_path', type=str)
    parser.add_argument('--out_path', type=str)
    args = parser.parse_args()
    main(args.gaussian_path, args.out_path)
