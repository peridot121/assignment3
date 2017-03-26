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


    red_wine_data_path = os.path.join(root, "data/wine/winequality-red-normalized-randomized-train.csv") # training data for 2015-02-04 to 2015-02-10 filtered to only unique values.
    red_wine_test_path = os.path.join(root, "data/wine/winequality-red-normalized-randomized-test.csv") # testing data for 2015-02-02 to 2015-02-04 (no overlap to training) also filtered to unique values.
    red_wine_data = pd.read_csv(red_wine_data_path, index_col=False).values
    red_wine_test = pd.read_csv(red_wine_test_path, index_col=False).values

    # Run clustering algorithms on unmodified data sets.
    run_clustering(transformed_census_data, transformed_census_test, "census")
    run_clustering(red_wine_data, red_wine_test, "redwine")

    # Run PCA and clustering on PCA output.
    pca_census_data, _ = run_pca(transformed_census_data, transformed_census_test, "census")
    pca_redwine_data, _ = run_pca(red_wine_data, red_wine_test, "redwine")
    if (pca_census_data is not None):
        run_clustering(pca_census_data, transformed_census_test, "census_pca")
    if (pca_redwine_data is not None):
        run_clustering(pca_redwine_data, red_wine_test, "redwine_pca")

    # Run ICA and clustering on ICA output.
    ica_census_data, _ = run_ica(transformed_census_data, transformed_census_test, "census")
    ica_redwine_data, _ = run_ica(red_wine_data, red_wine_test, "redwine")
    if (ica_census_data is not None):
        run_clustering(ica_census_data, transformed_census_test, "census_ica")
    if (ica_redwine_data is not None):
        run_clustering(ica_redwine_data, red_wine_test, "redwine_ica")

    # Run RP and clustering on RP output.
    rp_census_data, _ = run_rp(transformed_census_data, transformed_census_test, "census")
    rp_redwine_data, _ = run_rp(red_wine_data, red_wine_test, "redwine")
    if (rp_census_data is not None):
        run_clustering(rp_census_data, transformed_census_test, "census_rp")
    if (rp_redwine_data is not None):
        run_clustering(rp_redwine_data, red_wine_test, "redwine_rp")

    # Run RF and clustering on RF output.
    rf_census_data, _ = run_rf(transformed_census_data, transformed_census_test, "census")
    rf_redwine_data, _ = run_rf(red_wine_data, red_wine_test, "redwine")
    if (rf_census_data is not None):
        run_clustering(rf_census_data, transformed_census_test, "census_rf")
    if (rf_redwine_data is not None):
        run_clustering(rf_redwine_data, red_wine_test, "redwine_rf")

    print "Homework 3 automated data execution finished.\n"
