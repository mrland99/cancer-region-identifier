import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cluster_PCA(filename: str):
    """
    Clusters spatial transcriptomics data into two clusters by principal component.
    :param filename: file containing spatial transcriptomics data
    :return: 'n' by '3' pandas dataframe with spatial coordinates and cluster prediction for each cell
    """
    df = pd.read_csv(filename, sep="\t")
    spatial_data = df.iloc[:, 0]
    spatial_xy = []
    for spot in spatial_data:
        coordinates = spot.split('x')
        coordinates = [float(i) for i in coordinates]
        spatial_xy.append(coordinates)
    spatial_xy = pd.DataFrame(spatial_xy, columns=['x', 'y'])

    # includes data pre-processing step
    pca_components = principal_PCA(filename)

    # Cluster according to PCA
    model = KMeans(n_clusters=2)
    model.fit(pca_components)
    prediction = model.predict(pca_components)
    prediction = pd.DataFrame(prediction)

    spatial_xy['label'] = prediction
    return spatial_xy


def principal_PCA(filename: str):
    """
    :param filename: file containing spatial transcriptomics data
    :return: pandas dataframe of principal component of PCA analysis
    """
    df = pd.read_csv(filename, sep="\t")

    df = normalize_df(df)

    # Run PCA
    pca = PCA(n_components=1)
    pca_components = pca.fit_transform(df)

    # Save components to a DataFrame
    pca_components = pd.DataFrame(pca_components)

    return pca_components


def percent_dropout(filename: str):
    """

    :param filename: file containing spatial transcriptomics data
    :return: percent dropout component as a pandas dataframe
    """
    df = pd.read_csv(filename, sep="\t")
    spatial_data = df.iloc[:, 0]
    spatial_xy = []
    for spot in spatial_data:
        coordinates = spot.split('x')
        coordinates = [float(i) for i in coordinates]
        spatial_xy.append(coordinates)
    spatial_xy = pd.DataFrame(spatial_xy, columns=['x', 'y'])

    # add percent dropout
    df = normalize_df(df)
    percent_dropout_component = (df == 0).astype(int).sum(axis=1) / df.shape[0]

    return percent_dropout_component


def relabel_Spatial(spatial_xy, metric_vector, bound):
    """
    Relabels spatial_xy clusters based on metric_vector and choice of bound
    :param spatial_xy:
    :param metric_vector:
    :param bound:
    :return: Relabled spatial_xy
    """
    metric_vector = metric_vector.to_numpy()
    for i in range(len(spatial_xy)):
        if metric_vector[i] < bound:
            spatial_xy.loc[i, 'label'] = 0
        else:
            spatial_xy.loc[i, 'label'] = 1
    return spatial_xy


def normalize_df(df):
    """
    filters and preprocesses spatial transcriptomics data into dataframe
    :param df: pandas dataframe of spatial transcriptomics data
    :return:
    """
    df = df.drop(df.columns[0], axis=1)
    # filter genes with non-zero entries in less than 15 cells
    nonzero_col_sum = (df > 0).sum(axis=0)
    nonzero_col_sum.to_numpy()
    index = []
    for i in range(len(nonzero_col_sum)):
        if nonzero_col_sum[i] >= 15:
            index.append(i)
    df = df.iloc[:, index]

    sums = df.sum(axis=1)
    df = df.div(sums, axis=0)

    # Log normalize data
    df = df + 1
    df = np.log(df)

    scalar = StandardScaler()
    df = scalar.fit_transform(df)
    df = pd.DataFrame(df)
    return df


def plot(spatial_xy):
    """
    plots clustered spatial data
    :param spatial_xy: 'n' by '3' pandas dataframe with columns: |x|y|label|
    :return: nothing
    """
    # plot clustering
    sns.set_style('white')
    customPalette = ['#65c0ba', '#f76262']
    sns.set_palette(customPalette)
    sns.lmplot(data=spatial_xy, x='x', y='y', hue='label', fit_reg=False, legend=True, legend_out=True)
    plt.gca()


def from_PCA(filename):
    """
    Creates spatial_xy from .csv file containing spatial data + pca
    :param filename:
    :return: 
    """
    df = pd.read_csv(filename, header=None)
    spatial_data = df.iloc[:, 0]
    spatial_xy = []
    for spot in spatial_data:
        coordinates = spot.split('x')
        coordinates = [float(i) for i in coordinates]
        spatial_xy.append(coordinates)
    spatial_xy = pd.DataFrame(spatial_xy, columns=['x', 'y'])

    # no labels yet, set it all to -1
    spatial_xy['label'] = pd.DataFrame(np.full((len(spatial_xy), 1), -1))
    spatial_xy = round(spatial_xy).astype(int)

    return spatial_xy, df.iloc[:, 1]