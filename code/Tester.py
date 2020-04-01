from image import image_segmentation, image_transform, filter_for_tissue, convert_to_int
from similarityMetrics import match_labels, true_and_pred_labels
from cluster import cluster_PCA, principal_PCA, relabel_Spatial, plot, percent_dropout
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score, normalized_mutual_info_score


def test(layer_path, image_path):
    layer = layer_path
    image = image_path
    img = cv2.imread(image)
    segmented_image, labels = image_segmentation(img)
    spatial_xy = cluster_PCA(layer)
    spatial_xy = convert_to_int(spatial_xy)
    pca = principal_PCA(layer)
    spatial_xy = filter_for_tissue(spatial_xy, labels, -1, -1)

    # PCA threshold
    a = pca.to_numpy()

    bounds = np.linspace(min(a), max(a), num=200)
    r = []
    f = []
    n = []
    max_rand_score = -1
    spatial_xy_max = spatial_xy.copy()

    for i in range(len(bounds)):
        spatial_xy = relabel_Spatial(spatial_xy, pca, bounds[i])
        true, pred = true_and_pred_labels(spatial_xy, labels, -1, -1)
        rand = adjusted_rand_score(true, pred)
        if rand > max_rand_score:
            spatial_xy_max = spatial_xy.copy()
            max_rand_score = rand

        r.append(rand)
        f.append(f1_score(true, pred, pos_label=2))
        n.append(normalized_mutual_info_score(true, pred))

    plt.rcParams.update({'font.size': 12})

    p1 = plt.scatter(bounds, f)
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.title('F1 Score vs PCA Threshold')
    plt.show()
    print('Max F1 Score: ' + str(max(f)))

    p2 = plt.scatter(bounds, r)
    plt.xlabel('Threshold')
    plt.ylabel('ARI score')
    plt.title('ARI Score vs PCA Threshold')
    plt.show()
    print('Max ARI: Score ' + str(max(r)))

    p3 = plt.scatter(bounds, n)
    plt.xlabel('Threshold')
    plt.ylabel('NMI score')
    plt.title('NMI Score vs PCA Threshold')
    plt.show()
    print('Max NMI Score: ' + str(max(n)))

    plt.rcParams.update({'font.size': 22})

    plt.figure(figsize=(27, 27))
    plt.scatter(spatial_xy_max['x'] - 1, spatial_xy_max['y'] - 1, c=spatial_xy_max['label'], label=(0, 1))
    plt.imshow(segmented_image)
    plt.grid(b=True)
    plt.title('Best Cluster According to PCA Thresholding')
    plt.show()

    # Dropout Threshold
    dropout = percent_dropout(layer)

    bounds = np.linspace(0.45, 1, num=200)
    r = []
    f = []
    n = []
    max_rand_score = -1
    spatial_xy_max = spatial_xy.copy()

    for i in range(len(bounds)):
        spatial_xy = relabel_Spatial(spatial_xy, dropout, bounds[i])
        true, pred = true_and_pred_labels(spatial_xy, labels, -1, -1)
        rand = adjusted_rand_score(true, pred)
        if rand > max_rand_score:
            spatial_xy_max = spatial_xy.copy()
            max_rand_score = rand
        r.append(rand)
        f.append(f1_score(true, pred, pos_label=2))
        n.append(normalized_mutual_info_score(true, pred))

    plt.rcParams.update({'font.size': 12})

    p1 = plt.scatter(bounds, f)
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.title('F1 Score vs Percent Dropout Threshold')
    plt.show()
    print('Max F1 Score: ' + str(max(f)))

    p2 = plt.scatter(bounds, r)
    plt.xlabel('Threshold')
    plt.ylabel('ARI score')
    plt.title('ARI Score vs Precent Dropout Threshold')
    plt.show()
    print('Max ARI: Score ' + str(max(r)))

    p3 = plt.scatter(bounds, n)
    plt.xlabel('Threshold')
    plt.ylabel('NMI score')
    plt.title('NMI Score vs Precent Dropout Threshold')
    plt.show()
    print('Max NMI Score: ' + str(max(n)))

    plt.rcParams.update({'font.size': 22})

    plt.figure(figsize=(27, 27))
    plt.scatter(spatial_xy_max['x'] - 1, spatial_xy_max['y'] - 1, c=spatial_xy_max['label'], label=(0, 1))
    plt.imshow(segmented_image)
    plt.grid(b=True)
    plt.title('Best Cluster According to Precent Dropout Thresholding')
    plt.show()

