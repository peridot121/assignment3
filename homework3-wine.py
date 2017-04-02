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
from benchmark import run_baseline
from plotting import run_plotting

if __name__ == "__main__":
    pylab.rcParams['figure.figsize'] = 12, 9

    root = os.path.curdir

    red_wine_data_path = os.path.join(root, "data/wine/winequality-red-binary.csv")
    red_wine_data = pd.read_csv(red_wine_data_path, index_col=False).values

    # Run clustering algorithms on unmodified data sets.
    run_clustering(red_wine_data, "redwine")

    # Run PCA and clustering on PCA output.
    pca_redwine_data = run_pca(red_wine_data, "redwine")
    if (pca_redwine_data is not None):
        run_clustering(pca_redwine_data, "redwine_pca")

    # Run ICA and clustering on ICA output.
    ica_redwine_data = run_ica(red_wine_data, "redwine")
    if (ica_redwine_data is not None):
        run_clustering(ica_redwine_data, "redwine_ica")

    # Run RP and clustering on RP output.
    rp_redwine_data = run_rp(red_wine_data, "redwine")
    if (rp_redwine_data is not None):
        run_clustering(rp_redwine_data, "redwine_rp")

    # Run RF and clustering on RF output.
    rf_redwine_data = run_rf(red_wine_data, "redwine")
    if (rf_redwine_data is not None):
        run_clustering(rf_redwine_data, "redwine_rf")

    # Run a baseline NN for comparison.
    run_baseline(red_wine_data, "redwine")

    # Plot graphs.
    run_plotting("redwine")

    print "Homework 3 automated data execution finished.\n"
