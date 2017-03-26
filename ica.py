import os
import sys
import traceback
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM, nn_arch, nn_reg, random_state
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA


def run_ica(training_data, testing_data, name):
    """ Runs tests on Independent Component Analysis.
    """
    try:
        np.random.seed(0)

        root = os.path.curdir
        output_path = os.path.join(root, "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, name + "_ica_")

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
        ica = FastICA(random_state=random_state)
        tmp = ica.fit_transform(trainX)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0) + 3
        tmp.to_csv(out + 'scree.csv')

        # Data for step 2.
        dims = range(2, trainX.shape[1] + 1)
        grid = {'ica__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': network_shape}
        ica = FastICA(random_state=random_state)
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=random_state)
        pipe = Pipeline([('ica', ica), ('NN', mlp)])
        gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

        gs.fit(trainX, trainY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out + 'dim_red.csv')
        num_components = tmp.query('rank_test_score == 1')['param_ica__n_components'].values[0] # Take the number of components from the best result.
        print "Best # of components = {}".format(num_components)

        # Data for step 3.
        ica = FastICA(n_components=num_components, random_state=random_state)
        trainX2 = ica.fit_transform(trainX)
        train2 = pd.DataFrame(np.hstack((trainX2, np.atleast_2d(trainY).T)))
        cols = list(range(train2.shape[1]))
        cols[-1] = 'Result'
        train2.columns = cols
        # train2.to_hdf(out + 'datasets.hdf', name, complib='blosc', complevel=9)

        # TODO: Something with/for the test data.
        return train2.values, testing_data
    except Exception:
        print "\nError running ICA for {}.".format(name)
        print(traceback.format_exc())
        print(sys.exc_info()[0])
        return None, None