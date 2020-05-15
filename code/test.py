from analysis import label_components, match_labels, get_transformed_spatial_coordinates, principal_PCA, percent_dropout, get_precision_recall
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import cv2


def pca_analysis_results(image_file, data_file, true_labels_file, show_ari_plot=False, show_accuracy_plot=False,
                         show_nmi_plot=False, show_f1_plot=False, show_best_cluster=False, show_component_hist=False):
    # read in files
    layer = data_file
    true_labels = pd.read_csv(true_labels_file, delimiter="\t")
    true_labels = true_labels['label']
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get xy coordinates of spatial transcriptomics
    spatial_xy = get_transformed_spatial_coordinates(layer)

    # PCA threshold
    pca = principal_PCA(layer)
    a = pca.to_numpy()
    bounds = np.linspace(min(a), max(a), num=200)
    a = []
    r = []
    f = []
    n = []

    max_rand_score = -1
    true_max = []
    pred_max = []
    bound_max = 0
    precision = []
    recall = []
    for i in range(len(bounds)):
        pred_labels = label_components(pca[0], bounds[i])
        true = np.transpose(true_labels.to_numpy()).flatten()
        pred = np.transpose(pred_labels.to_numpy()).flatten()
        # pred, true = match_labels(pred_labels, true_labels)
        rand = adjusted_rand_score(true, pred)
        if rand > max_rand_score:
            spatial_xy_max = spatial_xy.copy()
            max_rand_score = rand
            true_max = true
            pred_max = pred
            bound_max = bounds[i]
        a.append(accuracy_score(true, pred))
        r.append(rand)
        f.append(f1_score(true, pred))
        n.append(normalized_mutual_info_score(true, pred))
        # precision_val, recall_val = get_precision_recall(true, pred)
        precision.append(precision_score(true, pred, zero_division=1))
        recall.append(recall_score(true, pred, zero_division=1))

    print('Threshold Value:' + str(bound_max))

    # seaborn histogram
    plt.rcParams.update({'font.size': 12})
    cancer = []
    noncancer = []
    for i in range(len(true_labels)):
        if true_labels[i] == 1:
            cancer.append(pca[0][i])
        else:
            noncancer.append(pca[0][i])

    if show_component_hist:
        sns.distplot(cancer, hist=False, kde=True,
                     bins=int(20), color='#90ee90',
                     hist_kws={'edgecolor': 'black'}, label="Cancerous")
        sns.distplot(noncancer, hist=False, kde=True,
                     bins=int(20), color='black',
                     hist_kws={'edgecolor': 'black'}, label="Non-cancerous")
        plt.axvline(bound_max, linestyle="--")
        plt.legend()

        # Add labels
        # plt.title('Density Plot of Princpal Components')
        plt.xlabel('PCA Components')
        plt.ylabel('Density')
        plt.show()

    if show_accuracy_plot:
        plt.scatter(bounds, f)
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy score')
        plt.title('Accuracy Score vs PCA Threshold')
        plt.show()
        print('Max Accuracy Score: ' + str(max(a)))

    if show_f1_plot:
        plt.scatter(bounds, f)
        plt.xlabel('Threshold')
        plt.ylabel('F1 score')
        plt.title('F1 Score vs PCA Threshold')
        plt.show()
        print('Max F1 Score: ' + str(max(f)))

    if show_ari_plot:
        plt.scatter(bounds, r)
        plt.xlabel('Threshold')
        plt.ylabel('ARI score')
        # plt.title('ARI Score vs PCA Threshold')
        plt.show()
        print('Max ARI: Score ' + str(max(r)))

    if show_nmi_plot:
        plt.scatter(bounds, n)
        plt.xlabel('Threshold')
        plt.ylabel('NMI score')
        plt.title('NMI Score vs PCA Threshold')
        plt.show()
        print('Max NMI Score: ' + str(max(n)))

    if show_best_cluster:
        plt.rcParams.update({'font.size': 48})

        plt.figure(figsize=(27, 27))
        colors = ['black', "#90ee90"]
        plt.scatter(spatial_xy['x'], spatial_xy['y'], c=pred_max, cmap=matplotlib.colors.ListedColormap(colors), s=250)
        plt.imshow(image)
        plt.grid(b=True)
        # plt.title('Best Cluster According to PCA Threshold')
        plt.show()

        # revert parameters back to normal
        plt.rcParams.update({'font.size': 12})

    return precision, recall


def percent_dropout_analysis_results(image_file, data_file, true_labels_file, show_ari_plot=False,
                                     show_accuracy_plot=False, show_nmi_plot=False, show_f1_plot=False,
                                     show_best_cluster=False, show_component_hist=False):
    # read in files
    layer = data_file
    true_labels = pd.read_csv(true_labels_file, delimiter="\t")
    true_labels = true_labels['label']
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get xy coordinates of spatial transcriptomics
    spatial_xy = get_transformed_spatial_coordinates(layer)

    # Percent dropout threshold
    percent = percent_dropout(layer)
    a = percent.to_numpy()
    bounds = np.linspace(min(a), max(a), num=200)
    a = []
    r = []
    f = []
    n = []

    max_rand_score = -1
    true_max = []
    pred_max = []
    precision = []
    recall = []
    bound_max = -1
    for i in range(len(bounds)):
        pred_labels = label_components(percent[0], bounds[i])
        true = np.transpose(true_labels.to_numpy()).flatten()
        pred = np.transpose(pred_labels.to_numpy()).flatten()
        # pred, true = match_labels(pred_labels, true_labels)
        rand = adjusted_rand_score(true, pred)
        if rand > max_rand_score:
            spatial_xy_max = spatial_xy.copy()
            max_rand_score = rand
            true_max = true
            pred_max = pred
            bound_max = bounds[i]
        a.append(accuracy_score(true, pred))
        r.append(rand)
        f.append(f1_score(true, pred))
        n.append(normalized_mutual_info_score(true, pred))
        precision.append(precision_score(true, pred, zero_division=1))
        recall.append(recall_score(true, pred, zero_division=1))

    # Plot Percent dropout Distribution
    plt.rcParams.update({'font.size': 12})
    # seaborn histogram
    cancer = []
    noncancer = []
    for i in range(len(true_labels)):
        if true_labels[i] == 1:
            cancer.append(percent[0][i])
        else:
            noncancer.append(percent[0][i])

    if show_component_hist:
        sns.distplot(cancer, hist=False, kde=True,
                     bins=int(20), color='#90ee90',
                     hist_kws={'edgecolor': 'black'}, label="Cancerous")
        sns.distplot(noncancer, hist=False, kde=True,
                     bins=int(20), color='black',
                     hist_kws={'edgecolor': 'black'}, label="Non-cancerous")
        plt.axvline(bound_max, linestyle="--")
        print(bound_max)
        plt.legend()


        # Add labels
        # plt.title('Density Plot of Percent Dropout Components')
        plt.xlabel('Percent Dropout Components')
        plt.ylabel('Density')
        plt.show()

    if show_accuracy_plot:
        plt.scatter(bounds, f)
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy score')
        plt.title('Accuracy Score vs Percent Dropout Threshold')
        plt.show()
        print('Max Accuracy Score: ' + str(max(a)))

    if show_f1_plot:
        plt.scatter(bounds, f)
        plt.xlabel('Threshold')
        plt.ylabel('F1 score')
        plt.title('F1 Score vs Percent Dropout Threshold')
        plt.show()
        print('Max F1 Score: ' + str(max(f)))

    if show_ari_plot:
        plt.scatter(bounds, r)
        plt.xlabel('Threshold')
        plt.ylabel('ARI score')
        # plt.title('ARI Score vs Percent Dropout Threshold')
        plt.show()
        print('Max ARI: Score ' + str(max(r)))

    if show_nmi_plot:
        plt.scatter(bounds, n)
        plt.xlabel('Threshold')
        plt.ylabel('NMI score')
        plt.title('NMI Score vs Percent Dropout Threshold')
        plt.show()
        print('Max NMI Score: ' + str(max(n)))

    if show_best_cluster:
        plt.rcParams.update({'font.size': 48})

        plt.figure(figsize=(27, 27))
        colors = ['black', "#90ee90"]
        plt.scatter(spatial_xy['x'], spatial_xy['y'], c=pred_max, cmap=matplotlib.colors.ListedColormap(colors), s=200)
        plt.imshow(image)
        plt.grid(b=True)
        # plt.title('Best Cluster According to Percent Dropout Threshold')
        plt.show()

    return precision, recall
