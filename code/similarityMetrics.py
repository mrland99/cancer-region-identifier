

def true_and_pred_labels(spatial_xy, labels,  dx=0, dy=0):
    """
    Returns the true labels and the predicted labels
    :param spatial_xy: spatial transcriptomics data with label
    :param labels: correct labels from image
    :param dx: x transform to align spatial data with image
    :param dy: y transform to align spatial data with image
    :return: true and predicted labels
    """
    spatial_predictions, sp_to_im_dict, im_to_sp_dict = match_labels(spatial_xy, labels, dx, dy)

    label_pred = []
    label_true = []
    for i in range(len(spatial_predictions)):
        x = spatial_predictions[i][0] + dx
        y = spatial_predictions[i][1] + dy
        label_true.append(labels[y][x])
        # convert spatial labels to image labels
        label_pred.append(sp_to_im_dict[spatial_predictions[i][2]])

    return label_true, label_pred


def match_labels(spatial_xy, labels, dx=0, dy=0):
    """
    Matches cluster labels between image and spatial
    :param spatial_xy: spatial transcriptomics data with label
    :param labels: correct labels from image
    :param dx: x transform to align spatial data with image
    :param dy: y transform to align spatial data with image
    :return: Filtered spatial_predictions, spatial to image label dict, image to spatial label dict
    """
    spatial_predictions = spatial_xy.to_numpy()

    # image background label (assuming tissue is in center)
    im_bg_label = labels[0][0]

    # figure out which image cluster label corresponds to which spatial cluster label
    # we know image labels are [0, 1, 2] and spatial are [0, 1]
    im_labels = [0, 1, 2]
    im_labels.remove(im_bg_label)
    im_label_a = im_labels[0]
    im_label_b = im_labels[1]
    sp_label_a = 0
    sp_label_b = 1

    # Two cases: im_label_a = sp_label_a,   im_label_a = sp_label_b
    # count number of matches
    aa_num_match = 0
    ab_num_match = 0
    for i in range(len(spatial_predictions)):
        x = spatial_predictions[i][0] + dx
        y = spatial_predictions[i][1] + dy
        im_label = labels[y][x]
        sp_label = spatial_predictions[i][2]
        if im_label == im_label_a:
            if sp_label == sp_label_a:
                aa_num_match += 1
            else:
                ab_num_match += 1
        else:
            if sp_label == sp_label_a:
                ab_num_match += 1
            else:
                aa_num_match += 1

    # create dictionary with proper label alignment
    im_to_sp_dict = {}
    sp_to_im_dict = {}
    if aa_num_match >= ab_num_match:
        im_to_sp_dict = {im_label_a: sp_label_a, im_label_b: sp_label_b}
        sp_to_im_dict = {sp_label_a: im_label_a, sp_label_b: im_label_b}
    else:
        im_to_sp_dict = {im_label_a: sp_label_b, im_label_b: sp_label_a}
        sp_to_im_dict = {sp_label_a: im_label_b, sp_label_b: im_label_a}

    return spatial_predictions, sp_to_im_dict, im_to_sp_dict
