# -*- coding: utf-8 -*-
import _init_paths
import os

import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import classification_report

from utils.config import Config
from utils.drawer import plot_confusion_matrix
from classifier import ClassifierOfflineTrain

def train_test_split(X, Y, ratio_of_test_size):
    ''' Split training data by ratio '''
    IS_SPLIT_BY_SKLEARN_FUNC = True

    # Use sklearn.train_test_split
    if IS_SPLIT_BY_SKLEARN_FUNC:
        RAND_SEED = 1
        tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
            X, Y, test_size=ratio_of_test_size, random_state=RAND_SEED)

    # Make train/test the same.
    else:
        tr_X = np.copy(X)
        tr_Y = Y.copy()
        te_X = np.copy(X)
        te_Y = Y.copy()
    return tr_X, te_X, tr_Y, te_Y

def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y):
    ''' Evaluate accuracy and time cost '''

    # Accuracy
    t0 = time.time()
    tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    print(f"Accuracy on training set is {tr_accu}")

    te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    print(f"Accuracy on testing set is {te_accu}")
    print("Accuracy report:")
    print(classification_report(
        te_Y, te_Y_predict, target_names=classes, output_dict=False))

    # Time cost
    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    print("Time cost for predicting one sample: "
          "{:.5f} seconds".format(average_time))

    # Plot accuracy
    axis, cf = plot_confusion_matrix(
        te_Y, te_Y_predict, classes, normalize=False, size=(12, 8))
    plt.show()


def main():
    # -- setting
    cfg = Config(config_file='../configs/train_action_recogn_pipeline.yaml')
    cfg_stage = cfg[os.path.basename(__file__)]
    classes = np.array(cfg.classes)

    get_path = lambda x: os.path.join(*x) if isinstance(x, (list, tuple)) else x
    src_features_X = get_path(cfg_stage.input.features_x)
    src_features_Y = get_path(cfg_stage.input.features_y)
    dst_model_path = cfg_stage.output.model_path

    # -- Load preprocessed data
    print("\nReading csv files of classes, features, and labels ...")
    X = np.loadtxt(src_features_X, dtype=float)  # features
    Y = np.loadtxt(src_features_Y, dtype=int)  # labels

    # -- Train-test split
    tr_X, te_X, tr_Y, te_Y = train_test_split(
        X, Y, ratio_of_test_size=0.3)
    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", len(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

    # -- Train the model
    print("\nStart training model ...")
    model = ClassifierOfflineTrain()
    model.train(tr_X, tr_Y)

    # -- Evaluate model
    print("\nStart evaluating model ...")
    evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y)

    # -- Save model
    print("\nSave model to " + dst_model_path)
    os.makedirs(os.path.dirname(dst_model_path), exist_ok=True)
    with open(dst_model_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
