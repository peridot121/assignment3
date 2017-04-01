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

    # Load census data
    # census_data_path = os.path.join(root, "data/census/combined.csv") # Full size dataset.
    census_data_path = os.path.join(root, "data/census/subsamplecombined.csv") # 20% size dataset.
    census_combined = pd.read_csv(census_data_path, index_col=False) # Need a combined data frame for label encoding to work properly.
    # Learned how to label encode from http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
    d = defaultdict(sk.preprocessing.LabelEncoder)
    transformed_combined_census_data = census_combined.apply(lambda x: d[x.name].fit_transform(x))
    census_values = transformed_combined_census_data.values
    # transformed_census_data = census_values[:32561,:] # 2/3 of data for training.
    # transformed_census_test = census_values[32561:,:] # 1/3 of data for cross validation.
    transformed_census_data = census_values[:6513,:] # 2/3 of data for training.
    transformed_census_test = census_values[6513:,:] # 1/3 of data for cross validation.

    # Run clustering algorithms on unmodified data sets.
    run_clustering(transformed_census_data, transformed_census_test, "census")

    # Run PCA and clustering on PCA output.
    pca_census_data, _ = run_pca(transformed_census_data, transformed_census_test, "census")
    if (pca_census_data is not None):
        run_clustering(pca_census_data, transformed_census_test, "census_pca")

    # Run ICA and clustering on ICA output.
    ica_census_data, _ = run_ica(transformed_census_data, transformed_census_test, "census")
    if (ica_census_data is not None):
        run_clustering(ica_census_data, transformed_census_test, "census_ica")

    # Run RP and clustering on RP output.
    rp_census_data, _ = run_rp(transformed_census_data, transformed_census_test, "census")
    if (rp_census_data is not None):
        run_clustering(rp_census_data, transformed_census_test, "census_rp")

    # Run RF and clustering on RF output.
    rf_census_data, _ = run_rf(transformed_census_data, transformed_census_test, "census")
    if (rf_census_data is not None):
        run_clustering(rf_census_data, transformed_census_test, "census_rf")

    # Run a baseline NN for comparison.
    run_baseline(transformed_census_data, transformed_census_test, "census")

    # Plot graphs.
    run_plotting("census")

    print "Homework 3 automated data execution finished.\n"
