# Adapted/copied from Jonathan Tay's code.
import os
import sys
import traceback





import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg, random_state
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def run_baseline(training_data, name):
    """ Runs a baseline Neural Net on the data.
    """
    try:
        np.random.seed(42)

        root = os.path.curdir
        output_path = os.path.join(root, "output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        out = os.path.join(output_path, name)

        trainX = training_data[:, :-1]
        trainY = training_data[:, training_data.shape[1] - 1]

        # Scale to [0, 1]
        trainX = StandardScaler().fit_transform(trainX)


        # network_shape = list(nn_arch)
        # network_shape.append((training_data.shape[1] / 2,))
        network_shape = list((training_data.shape[1] / 2,))

        # Benchmarking for chart type 2
        grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':network_shape}
        mlp = MLPClassifier(activation='relu',max_iter=3000,early_stopping=True,random_state=random_state)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

        gs.fit(trainX,trainY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out + 'baseline.csv')
    except Exception:
        print "\nError running the baseline for {}.".format(name)
        print(traceback.format_exc())
        print(sys.exc_info()[0])
        return None, None
