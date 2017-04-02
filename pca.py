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
from sklearn.decomposition import PCA

def run_pca(training_data, name):
    """ Runs tests on Principal Component Analysis.
    """
    try:
        np.random.seed(42)

        root = os.path.curdir
        output_path = os.path.join(root, "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, name + "_pca_")

        trainX = training_data[:, :-1]
        trainY = training_data[:, training_data.shape[1] - 1]

        # Scale to [0, 1]
        trainX = StandardScaler().fit_transform(trainX)

        # network_shape = list(nn_arch)
        # network_shape.append((training_data.shape[1] / 2,))
        network_shape = list((training_data.shape[1] / 2,))

        # Data for step 1.
        pca = PCA(random_state=random_state)
        pca.fit(trainX)
        index_range = max([len(pca.explained_variance_), min([len(pca.explained_variance_), 500])]) + 1
        tmp = pd.Series(data=pca.explained_variance_, index=range(1, index_range))
        tmp.to_csv(out + 'scree.csv')

        # Data for step 2.
        dims = range(2, trainX.shape[1] + 1)
        grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':network_shape}
        pca = PCA(random_state=random_state)
        mlp = MLPClassifier(activation='relu',max_iter=3000,early_stopping=True,random_state=random_state)
        pipe = Pipeline([('pca',pca),('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,n_jobs=-1,solver='lbfgs')

        gs.fit(trainX,trainY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out + 'dim_red.csv')
        num_components = tmp.query('rank_test_score == 1')['param_pca__n_components'].values[0] # Take the number of components from the best result.
        print "Best # of components = {}".format(num_components)

        # Data for step 3.
        # Transform and save the results for later.
        pca = PCA(n_components=num_components, random_state=random_state)
        trainX2 = pca.fit_transform(trainX)
        train2 = pd.DataFrame(np.hstack((trainX2,np.atleast_2d(trainY).T)))
        cols = list(range(train2.shape[1]))
        cols[-1] = 'Result'
        train2.columns = cols

        return train2.values
    except Exception:
        print "\nError running PCA for {}.".format(name)
        print(traceback.format_exc())
        print(sys.exc_info()[0])
        return None, None
