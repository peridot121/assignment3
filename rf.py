# Adapted/copied from Jonathan Tay's code.
import os
import sys
import traceback
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg, ImportanceSelect, random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection


def run_rf(training_data, testing_data, name):
    """ Runs tests on Random Forest.
    """
    try:
        np.random.seed(0)

        root = os.path.curdir
        output_path = os.path.join(root, "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, name + "_rf_")

        testX = testing_data[:, :-1]
        testY = testing_data[:, testing_data.shape[1] - 1]
        trainX = training_data[:, :-1]
        trainY = training_data[:, training_data.shape[1] - 1]

        # Scale to [0, 1]
        scaler = StandardScaler()
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)

        network_shape = list(nn_arch)
        network_shape.append((training_data.shape[1] / 2,))

        # Data for step 1.
        rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state, n_jobs=7)
        fs_train = rfc.fit(trainX, trainY).feature_importances_

        tmp = pd.Series(np.sort(fs_train)[::-1])
        tmp.to_csv(out + 'scree.csv')

        # Data for step 2.
        filtr = ImportanceSelect(rfc)
        dims = range(2, trainX.shape[1] + 1)
        grid = {'filter__n': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': network_shape}
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=random_state)
        pipe = Pipeline([('filter', filtr), ('NN', mlp)])
        gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

        gs.fit(trainX, trainY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out + 'dim_red.csv')

        # Data for step 3.
        # Set this from chart 2 and dump, use clustering script to finish up
        filtr = ImportanceSelect(rfc)

        trainX2 = filtr.fit_transform(trainX, trainY)
        train2 = pd.DataFrame(np.hstack((trainX2, np.atleast_2d(trainY).T)))
        cols = list(range(train2.shape[1]))
        cols[-1] = 'Result'
        train2.columns = cols
        # train2.to_hdf(out + 'datasets.hdf', name, complib='blosc', complevel=9)

        # TODO: Something with/for the test data.
        return train2.values, testing_data
    except Exception:
        print "\nError running RF for {}.".format(name)
        print(traceback.format_exc())
        print(sys.exc_info()[0])
        return None, None
