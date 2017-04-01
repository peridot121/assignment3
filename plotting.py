import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

def run_plotting(name):
    root = os.path.curdir
    datadir = os.path.join(root, 'output')
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    chartsdir = os.path.join(root, 'charts')
    if not os.path.exists(chartsdir):
        os.makedirs(chartsdir)

    # Run for plain clustering, Principal Component Analysis, Independent Component Analysis, Random Projection, and Random Forest
    algonames = ['', 'pca', 'ica', 'rp', 'rf']
    # clusterdatanames = ['SSE', 'logliklihood', 'acc', 'adjMI', 'Kmeans', 'GMM', '2D']
    capitalname = name.capitalize()
    param_names = ['param_pca__n_components', 'param_ica__n_components', 'param_rp__n_components', 'param_filter__n']

    for index, algo in enumerate(algonames):
        # Setup file paths
        filenamefragment = '{0}_{1}{2}cluster_'.format(name, algo, '_' if len(algo) > 0 else '')
        datain = os.path.join(datadir, filenamefragment)
        out = os.path.join(chartsdir, filenamefragment)

        # SSE
        df = pd.read_csv(datain + 'SSE.csv', index_col=0)
        displayname = '{0} {1} '.format(capitalname, algo.upper())
        plot_data(df, ['bx-'], displayname + 'SSE vs. clusters', "Clusters", "SSE", out + 'SSE.png')

        # Log likelihood
        df = pd.read_csv(datain + 'logliklihood.csv', index_col=0)
        displayname = '{0} {1} '.format(capitalname, algo.upper())
        plot_data(df, ['bx-'], displayname + 'Log likelihood vs. clusters', "Clusters", "Log likelihood", out + 'Log-likelihood.png', loc='lower right')

        # Cluster accuracy classification score
        df = pd.read_csv(datain + 'acc.csv', index_col=0)
        displayname = '{0} {1} '.format(capitalname, algo.upper())
        plot_data(df.T, None, displayname + 'Cluster Accuracy Classification Score', "Clusters", "Accuracy", out + 'accuracy.png', loc='lower right')

        # Adjusted Mutual Information
        df = pd.read_csv(datain + 'adjMI.csv', index_col=0)
        displayname = '{0} {1} '.format(capitalname, algo.upper())
        plot_data(df.T, None, displayname + 'Adjusted Mutual Information Score', "Clusters", "Adjusted MI Score", out + 'AdjustedMutualInfo.png', loc='lower right')

        # Cluster
        clusterdf = []
        for ty in ['adjMI','acc']:
            df = pd.read_csv(datain + ty + '.csv', index_col=0).T
            cols = list(df.columns)
            cols = [col + ' ' + ty for col in cols]
            df.columns = cols
            clusterdf.append(df.copy())

        clusterdf = pd.concat(clusterdf,1)
        clusterdf.index = clusterdf.index.astype(int)
        clusterdf = clusterdf.rename(columns = lambda x: x.replace('adjMI','Adj. MI').replace('acc','Accuracy'))

        shorthand = {'GMM':'param_gmm__n_components','Kmeans':'param_km__n_clusters'}
        for clust in ['GMM','Kmeans']:
            df = pd.read_csv(datain + clust +'.csv',index_col=0)[[shorthand[clust],'mean_test_score']]
            df = df.groupby(shorthand[clust]).max()
            df.columns = [clust]
            clusterdf = pd.concat([clusterdf, df], 1)
        plot_data(clusterdf, ['b--','g--','b-.','g-.','b-','g-'], '{} Best NN CV Accuracy.'.format(capitalname), "Clusters", outfile = out + 'ClusterScores.png', loc='center right')

        clusterdf = pd.DataFrame([])
        shorthand = {'GMM':'param_gmm__n_components','Kmeans':'param_km__n_clusters'}
        for clust in ['GMM','Kmeans']:
            df = pd.read_csv(datain + clust +'.csv',index_col=0)[[shorthand[clust],'mean_test_score']]
            df = df.groupby(shorthand[clust]).max()
            df.columns = [clust]
            clusterdf = pd.concat([clusterdf, df], 1)
        plot_data(clusterdf, ['b-', 'g-'], '{} Best NN CV Accuracy.'.format(capitalname), "Clusters", outfile = out + 'ClusterScores2.png', loc='center right')

        # Cluster scatter plot t-SNE
        df = pd.read_csv(datain + '2D.csv', index_col=0, dtype={'target':int})
        displayname = '{0} {1} '.format(capitalname, algo.upper())
        ax = df.plot(title=displayname + 't-SNE 2D Class Targets', fontsize=12, kind="scatter", x='x', y='y', c='target', cmap='cool')
        # ax.legend(loc = 'upper right')
        plt.savefig(out + 'ScatterTSNE.png', bbox_inches='tight')
        plt.close()

        # Correct cluster plot?
        df = pd.read_csv(datain + '2DCluster.csv', index_col=0, dtype={'target':int, 'KmeansCluster':int, 'GMMCluster':int})
        cluster_algos = ['GMM','Kmeans']
        for clustindex, clust in enumerate(cluster_algos):
            vals = df.drop(cluster_algos[1-clustindex] + 'Cluster', 1, inplace=False)
            ax = vals.plot(title=displayname + clust + ' t-SNE 2D Clusters', fontsize=12, kind="scatter", x='x', y='y', c=clust+'Cluster', cmap='Paired')
            # ax = vals.plot(title=displayname + clust + ' t-SNE 2D Clusters', fontsize=12, kind="scatter", x='x', y='y', c=clust+'Cluster', cmap='gist_rainbow')
            # ax = vals.plot(title=displayname + clust + ' t-SNE 2D Clusters', fontsize=12, kind="scatter", x='x', y='y', c=clust+'Cluster', cmap='nipy_spectral')
#            ax.legend(loc = 'upper right')
            plt.savefig(out + clust + 'Clusters.png', bbox_inches='tight')
            plt.close()

        # NN Grid Search Dimension Reduction
        if (algo != ''):
            dimredpath = '{0}_{1}_dim_red.'.format(name, algo)
            groupbyparam = param_names[index - 1]
            df = pd.read_csv(os.path.join(datadir, dimredpath + 'csv'),index_col=0)
            s = df.groupby(groupbyparam)['mean_test_score'].max()
            plot_data(s, None, 'Best NN CV Accuracy', 'Using Top N Features', 'Accuracy', os.path.join(chartsdir, dimredpath + 'png'), 'center right')

    # PCA specific
    datadirname = os.path.join(datadir, name)
    chartsdirname = os.path.join(chartsdir, name)
    df = pd.read_csv(datadirname + '_pca_scree.csv', index_col=0)
    plot_data(df, ['bx-'], capitalname + " PCA Explained Variance", "Components", "Variance", chartsdirname + '_pca_variance.png', label="Variance")

    # ICA specific
    df = pd.read_csv(datadirname + '_ica_scree.csv', index_col=0)
    plot_data(df, ['bx-'], capitalname + " ICA (normalized) Kurtosis", "Components", "(normalized) Kurtosis", chartsdirname + '_ica_kurtosis.png', label="Kurtosis")

    # RP specific
    df = pd.read_csv(datadirname + '_rp_scree.csv', index_col=0)
    df = df.mean(axis=1)
    df2 = pd.read_csv(datadirname + '_rp_scree2.csv', index_col=0)
    df2 = df2.mean(axis=1)
    df = pd.concat([df, df2], 1)
    df.columns = ['Distance Correlation', 'Reconstruction Error']
    # print df
    plot_data(df, None, capitalname + " RP Quality", "Projections", "", chartsdirname + '_rp_quality.png', 'center right')

    # RF specific
    df = pd.read_csv(datadirname + '_rf_scree.csv', index_col=0)
    plot_data(df, ['bx-'], capitalname + " RF feature importances", "", "Importance %", chartsdirname + '_rf_feature_importance.png', label="Importance")

def plot_data(df, style=None, title="", xlabel=None, ylabel=None, outfile = '', loc='upper right', label=None):
    ax = df.plot(title=title, fontsize=12, style=style, label=label)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.legend(loc=loc, framealpha=0.5)

    if outfile == '':
        plt.show()
    else:
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    # plt.clf() # Prevent getting merged data.

if __name__ == "__main__":
    # pylab.rcParams['figure.figsize'] = 16, 12
    pylab.rcParams['figure.figsize'] = 12, 9

    # Generate pretty graphs
    run_plotting("census")
    run_plotting("redwine")
    plt.close('all')
