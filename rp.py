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
from helpers import cluster_acc, myGMM, pairwiseDistCorr, nn_reg, nn_arch, reconstructionError, random_state
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection
from itertools import product

def run_rp(training_data, name):
    """ Runs tests on Random Projection.
    """
    try:
        np.random.seed(42)

        root = os.path.curdir
        output_path = os.path.join(root, "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, name + "_rp_")

        trainX = training_data[:, :-1]
        trainY = training_data[:, training_data.shape[1] - 1]

        # Scale to [0, 1]
        trainX = StandardScaler().fit_transform(trainX)

        # network_shape = list(nn_arch)
        # network_shape.append((training_data.shape[1] / 2,))
        network_shape = list((training_data.shape[1] / 2,))
        dims = range(2, trainX.shape[1] + 1)

        # Data for step 1.
        tmp = defaultdict(dict)
        for i, dim in product(range(10), dims):
            rp = SparseRandomProjection(random_state=i, n_components=dim)
            tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(trainX), trainX)
        tmp =pd.DataFrame(tmp).T
        tmp.to_csv(out + 'scree.csv')

        # Reconstruction error.
        tmp = defaultdict(dict)
        for i, dim in product(range(10), dims):
            rp = SparseRandomProjection(random_state=i, n_components=dim)
            rp.fit(trainX)
            tmp[dim][i] = reconstructionError(rp, trainX)
        tmp =pd.DataFrame(tmp).T
        tmp.to_csv(out + 'scree2.csv')

        # Data for step 2.
        grid ={'rp__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':network_shape}
        rp = SparseRandomProjection(random_state=random_state)       
        mlp = MLPClassifier(activation='relu',max_iter=3000,early_stopping=True,random_state=random_state,solver='lbfgs')
        pipe = Pipeline([('rp',rp),('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,n_jobs=-1)

        gs.fit(trainX,trainY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out + 'dim_red.csv')
        num_components = tmp.query('rank_test_score == 1')['param_rp__n_components'].values[0] # Take the number of components from the best result.
        print "Best # of components = {}".format(num_components)

        # Data for step 3.
        rp = SparseRandomProjection(n_components=num_components,random_state=random_state)

        trainX2 = rp.fit_transform(trainX)
        train2 = pd.DataFrame(np.hstack((trainX2,np.atleast_2d(trainY).T)))
        cols = list(range(train2.shape[1]))
        cols[-1] = 'Result'
        train2.columns = cols

        return train2.values
    except Exception:
        print "\nError running RP for {}.".format(name)
        print(traceback.format_exc())
        print(sys.exc_info()[0])
        return None, None
