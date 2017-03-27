import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import sklearn as sk
from collections import defaultdict

from clustering import run_clustering
from pca import run_pca
from ica import run_ica
from rp import run_rp
from rf import run_rf

if __name__ == "__main__":
    pylab.rcParams['figure.figsize'] = 16, 12

    root = os.path.curdir

    red_wine_data_path = os.path.join(root, "data/wine/winequality-red-normalized-randomized-binary-train.csv") # training data for 2015-02-04 to 2015-02-10 filtered to only unique values.
    red_wine_test_path = os.path.join(root, "data/wine/winequality-red-normalized-randomized-binary-test.csv") # testing data for 2015-02-02 to 2015-02-04 (no overlap to training) also filtered to unique values.
    red_wine_data = pd.read_csv(red_wine_data_path, index_col=False).values
    red_wine_test = pd.read_csv(red_wine_test_path, index_col=False).values

    # Run clustering algorithms on unmodified data sets.
    run_clustering(red_wine_data, red_wine_test, "redwine")

    # Run PCA and clustering on PCA output.
    pca_redwine_data, _ = run_pca(red_wine_data, red_wine_test, "redwine")
    if (pca_redwine_data is not None):
        run_clustering(pca_redwine_data, red_wine_test, "redwine_pca")

    # Run ICA and clustering on ICA output.
    ica_redwine_data, _ = run_ica(red_wine_data, red_wine_test, "redwine")
    if (ica_redwine_data is not None):
        run_clustering(ica_redwine_data, red_wine_test, "redwine_ica")

    # Run RP and clustering on RP output.
    rp_redwine_data, _ = run_rp(red_wine_data, red_wine_test, "redwine")
    if (rp_redwine_data is not None):
        run_clustering(rp_redwine_data, red_wine_test, "redwine_rp")

    # Run RF and clustering on RF output.
    rf_redwine_data, _ = run_rf(red_wine_data, red_wine_test, "redwine")
    if (rf_redwine_data is not None):
        run_clustering(rf_redwine_data, red_wine_test, "redwine_rf")

    print "Homework 3 automated data execution finished.\n"