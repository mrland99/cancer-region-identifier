import cv2
import numpy as np
import pandas as pd


def file_to_image(filename: str):
    # read the image
    image = cv2.imread(filename)
    return image


def image_segmentation(image):
    # transform image to proper dimension and alignment
    image = image_transform(image)

    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))

    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # reshape labels back to original image dimension.
    # image.shape returns (height, width, channel)
    cluster_labels = labels.reshape(image.shape[:2])

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image, cluster_labels


def image_transform(image):
    # resize takes (width, height)
    crop_img = cv2.resize(image, (32, 34), interpolation=cv2.INTER_AREA)
    return crop_img


def filter_for_tissue(spatial_xy, im_labels, dx=0, dy=0):
    """
    Filters out invidual samples that do not match up with tissue in image
    :param spatial_xy: spatial transcriptomics data with label
    :param im_labels: correct labels from image
    :param dx: x transform to align spatial data with image
    :param dy: y transform to align spatial data with image
    :return: filtered spatial_xy
    """
    spatial_xy = spatial_xy.to_numpy()

    # image background label (assuming tissue is in center)
    im_bg_label = im_labels[0][0]

    # remove all spatial_prediction samples that lie on background (does not align up with actual tissue sample)
    delete_set = []
    for i in range(len(spatial_xy)):
        x = spatial_xy[i][0] + dx
        y = spatial_xy[i][1] + dy
        # Remember, x --> col, y --> row
        if im_labels[y][x] == im_bg_label:
            delete_set.append(i)
    filtered_spatial_xy = np.delete(spatial_xy, delete_set, axis=0)
    # convert back to dataframe
    filtered_spatial_xy = pd.DataFrame(filtered_spatial_xy)
    filtered_spatial_xy.columns = ['x', 'y', 'label']
    return filtered_spatial_xy


def convert_to_int(spatial_xy):
    """
    Converts coordiates of spatial_xy to ints
    :param spatial_xy:
    :return:
    """
    return round(spatial_xy).astype(int)
