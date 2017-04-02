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


def run_rf(training_data, name):
    """ Runs tests on Random Forest.
    """
    try:
        np.random.seed(42)

        root = os.path.curdir
        output_path = os.path.join(root, "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, name + "_rf_")

        trainX = training_data[:, :-1]
        trainY = training_data[:, training_data.shape[1] - 1]

        # Scale to [0, 1]
        trainX = StandardScaler().fit_transform(trainX)

        # network_shape = list(nn_arch)
        # network_shape.append((training_data.shape[1] / 2,))
        network_shape = list((training_data.shape[1] / 2,))

        # Data for step 1.
        rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state, n_jobs=7)
        fs_train = rfc.fit(trainX, trainY).feature_importances_

        tmp = pd.Series(np.sort(fs_train)[::-1])
        tmp.to_csv(out + 'scree.csv')

        # Data for step 2.
        filtr = ImportanceSelect(rfc)
        dims = range(2, trainX.shape[1] + 1)
        grid = {'filter__n': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': network_shape}
        mlp = MLPClassifier(activation='relu', max_iter=3000, early_stopping=True, random_state=random_state)
        pipe = Pipeline([('filter', filtr), ('NN', mlp)])
        gs = GridSearchCV(pipe, grid, verbose=10, cv=5,n_jobs=-1,solver='lbfgs')

        gs.fit(trainX, trainY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out + 'dim_red.csv')

        # Data for step 3.
        # Set this from chart 2 and dump, use clustering script to finish up
        dim = tmp.query('rank_test_score == 1')['param_filter__n'].values[0] # Take the 'n' from the best result.
        print "Best 'n' = {}".format(dim)
        filtr = ImportanceSelect(rfc, dim)

        trainX2 = filtr.fit_transform(trainX, trainY)
        train2 = pd.DataFrame(np.hstack((trainX2, np.atleast_2d(trainY).T)))
        cols = list(range(train2.shape[1]))
        cols[-1] = 'Result'
        train2.columns = cols

        return train2.values
    except Exception:
        print "\nError running RF for {}.".format(name)
        print(traceback.format_exc())
        print(sys.exc_info()[0])
        return None, None
