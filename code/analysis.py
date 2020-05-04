import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_transformed_spatial_coordinates(filename: str):
    df = pd.read_csv(filename, sep="\t")
    spatial_data = df.iloc[:, 0]
    spatial_xy = []
    for spot in spatial_data:
        coordinates = spot.split('x')
        coordinates = [float(i) for i in coordinates]
        spatial_xy.append(coordinates)
    xy_coordinates = pd.DataFrame(spatial_xy, columns=['x', 'y'])

    # transform image
    x_scale = 288.9
    y_scale = 292.6
    x_shift = -288.9
    y_shift = -292.6

    xy_coordinates['x'] = xy_coordinates['x'] * x_scale + x_shift
    xy_coordinates['y'] = xy_coordinates['y'] * y_scale + y_shift
    return xy_coordinates


def label_components(pca, bound):
    components = pca.to_numpy(copy=True)
    for i in range(len(pca)):
        if components[i] < bound:
            components[i] = 1
        else:
            components[i] = 0
    pca_components = pd.DataFrame(components).astype(int)
    return pca_components


def match_labels(pred_labels, true_labels):
    """
    true and pred labels are both 0, 1. Analyzes and matches which label in true corresponds to which label in true.
    Then adjusts labels so they align.
    :param true_labels: Set of true labels as column data frame
    :param pred_labels: Set of pred labels as column data frame
    :return: correctly labeled true and predicted labels as lists
    """
    true_np = true_labels.to_numpy(copy=True)
    pred_np = pred_labels.to_numpy(copy=True)
    # count number of 0-0 and 1-1 matches
    same_count = 0
    for i in range(len(true_np)):
        if true_np[i] == pred_np[i]:
            same_count += 1

    # If over half are 0-0 and 1-1 labels, its probably correct. Otherwise, swap 0 and 1 in pred
    if same_count < (len(true_np) / 2):
        for i in range(len(pred_np)):
            if pred_np[i] == 1:
                pred_np[i] = 0
            else:
                pred_np[i] = 1

    return np.transpose(pred_np).flatten(), np.transpose(true_np).flatten()


def percent_dropout(filename: str):
    """

    :param filename: file containing spatial transcriptomics data
    :return: percent dropout component as a pandas dataframe
    """
    df = pd.read_csv(filename, sep="\t")
    df = df.drop(df.columns[0], axis=1)
    percent_dropout_component = pd.DataFrame((df == 0).astype(int).sum(axis=1) / df.shape[0])

    return percent_dropout_component


def normalize_df(df):
    """
    filters and preprocesses spatial transcriptomics data into dataframe
    :param df: pandas dataframe of spatial transcriptomics data
    :return:
    """
    df = df.drop(df.columns[0], axis=1)
    # filter genes with at least 15 non-zero entries
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


def principal_PCA(filename: str):
    """
    :param filename: file containing spatial transcriptomics data
    :return: pandas dataframe of principal component of PCA analysis
    """
    df = pd.read_csv(filename, sep="\t")

    df = normalize_df(df)

    # Run PCA
    pca = PCA(n_components=1, svd_solver='full')
    pca_components = pca.fit_transform(df)

    # Save components to a DataFrame
    pca_components = pd.DataFrame(pca_components)

    return pca_components


def get_precision_recall(true, pred):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(true)):
        if true[i] == 1 and pred[i] == 1:
            tp = tp + 1
        if true[i] == 0 and pred[i] == 1:
            fp = fp + 1
        if true[i] == 0 and pred[i] == 1:
            fn = fn + 1
        if true[i] == 0 and pred[i] == 0:
            tn = tn + 1

    if (tp + fp) == 0:
        precision = 1
    else:
        precision = tp / (tp + fp)
    if (fp + fn) != 0:
        recall = tp / (fp + fn)
    else:
        recall = 1
    return precision, recall
