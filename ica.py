# Adapted/copied from Jonathan Tay's code.
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


def run_ica(training_data, name):
    """ Runs tests on Independent Component Analysis.
    """
    try:
        np.random.seed(42)

        root = os.path.curdir
        output_path = os.path.join(root, "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, name + "_ica_")

        trainX = training_data[:, :-1]
        trainY = training_data[:, training_data.shape[1] - 1]

        # Scale to [0, 1]
        trainX = StandardScaler().fit_transform(trainX)

        # network_shape = list(nn_arch)
        # network_shape.append((training_data.shape[1] / 2,))
        network_shape = list((training_data.shape[1] / 2,))

        # Data for step 1.
        dims = range(2, trainX.shape[1] + 1)
        ica = FastICA(random_state=random_state)
        kurt = {}
        for dim in dims:
            ica.set_params(n_components=dim)
            tmp = ica.fit_transform(trainX)
            tmp = pd.DataFrame(tmp)
            tmp = tmp.kurt(axis=0)
            kurt[dim] = tmp.abs().mean()

        kurt = pd.Series(kurt)
        kurt.to_csv(out + 'scree.csv')

        # Data for step 2.
        grid = {'ica__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': network_shape}
        ica = FastICA(random_state=random_state)
        mlp = MLPClassifier(activation='relu', max_iter=3000, early_stopping=True, random_state=random_state)
        pipe = Pipeline([('ica', ica), ('NN', mlp)])
        gs = GridSearchCV(pipe, grid, verbose=10, cv=5,n_jobs=-1,solver='lbfgs')

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

        return train2.values
    except Exception:
        print "\nError running ICA for {}.".format(name)
        print(traceback.format_exc())
        print(sys.exc_info()[0])
        return None, None
