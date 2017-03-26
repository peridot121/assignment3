# Adapted/copied from Jonathan Tay's code.
import os
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
import sys
import traceback

def run_clustering(training_data, testing_data, name):
    """ Runs tests on K-Means clustering.
    """
    try:
        np.random.seed(0)

        root = os.path.curdir
        output_path = os.path.join(root, "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, name + "_cluster_")

        # if (training_data == None or testing_data == None):
        #     training_data = pd.read_hdf(os.path.join(output_path, name + 'datasets.hdf'), name)
        #     testing_data =  pd.read_hdf(os.path.join(output_path, name + 'datasets_test.hdf'), name)
        testX = testing_data[:, :-1]
        testY = testing_data[:, testing_data.shape[1] - 1]
        trainX = training_data[:, :-1]
        trainY = training_data[:, training_data.shape[1] - 1]

        # Scale to [0, 1]
        scaler = StandardScaler()
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        # testX = scaler.transform(testX) # This won't work if the training data was transformed by a prior operation such as PCA, ICA, or RP but the test data is unmodified.

        network_shape = list(nn_arch)
        network_shape.append((training_data.shape[1] / 2,))

        SSE = defaultdict(dict)
        ll = defaultdict(dict)
        acc = defaultdict(lambda: defaultdict(dict))
        adjMI = defaultdict(lambda: defaultdict(dict))
        km = kmeans(random_state=random_state)
        gmm = GMM(random_state=random_state)

        data_clusters = range(2, testing_data.shape[1] + 1)
        st = clock()
        for k in data_clusters:
            km.set_params(n_clusters=k)
            gmm.set_params(n_components=k)
            km.fit(trainX)
            gmm.fit(trainX)
            SSE[k][name] = km.score(trainX)
            ll[k][name] = gmm.score(trainX)
            acc[k][name]['Kmeans'] = cluster_acc(trainY, km.predict(trainX))
            acc[k][name]['GMM'] = cluster_acc(trainY, gmm.predict(trainX))
            adjMI[k][name]['Kmeans'] = ami(trainY, km.predict(trainX))
            adjMI[k][name]['GMM'] = ami(trainY, gmm.predict(trainX))
            print(k, clock() - st)

        SSE = (-pd.DataFrame(SSE)).T
        SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
        ll = pd.DataFrame(ll).T
        ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)
        acc = pd.Panel(acc)
        adjMI = pd.Panel(adjMI)

        SSE.to_csv(out + 'SSE.csv')
        ll.to_csv(out + 'logliklihood.csv')
        acc.ix[:, :, name].to_csv(out + 'acc.csv')
        adjMI.ix[:, :, name].to_csv(out + 'adjMI.csv')

        # Grid search NN with clusters.
        grid ={'km__n_clusters':data_clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':network_shape}
        mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=random_state)
        km = kmeans(random_state=random_state)
        pipe = Pipeline([('km',km),('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10)

        gs.fit(trainX,trainY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out + 'Kmeans.csv')

        grid ={'gmm__n_components':data_clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':network_shape}
        mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=random_state)
        gmm = myGMM(random_state=random_state)
        pipe = Pipeline([('gmm',gmm),('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

        gs.fit(trainX,trainY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out + 'GMM.csv')

        trainX2D = TSNE(verbose=10,random_state=random_state).fit_transform(trainX)
        data2D = pd.DataFrame(np.hstack((trainX2D,np.atleast_2d(trainY).T)),columns=['x','y','target'])
        data2D.to_csv(out + '2D.csv')
    except Exception:
        print "\nError running clustering for {}.".format(name)
        print(traceback.format_exc())
        print(sys.exc_info()[0])
