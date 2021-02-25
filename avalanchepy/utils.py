import numpy as np

from skimage.measure import find_contours
import scipy


def get_contours_ini(phabin):
    """Returns the contours of input array The contours follow the pixel edges instead of cutting through them.

    Arguments:
    ---------
    phabin: binarised phase image of the initial domain wall configuration

    Returns:
    -------
    contours: an array of contour coordinates following the pixel edges

    """

    # Scale phabin by a factor of 2
    tmp_zoomed = np.kron(phabin != 0, np.ones((2, 2)))
    contours_zoomed = find_contours(tmp_zoomed, 0.5)
    contours = [np.ceil(x / 2.) - 0.5 for x in contours_zoomed]

    return contours


def get_contours_full(switchmap, cutoff_time, struc_elem=True):
    """Extracts the contours of each switching scan in switchmap. The contours follow the pixel edges.

    Arguments:
    ---------
    switchmap: map of switching scans
    cutoff_time: cutoff value for switchmap if we use only part of the measurement series

    Keyword arguments:
    -----------------
    struc_elem: structuring element for the labeling function (default True)

    Returns:
    -------
    contours: an array of contours, with element i corresponding to areas where switchmap=i

    """

    structuring_element = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    contours = []
    for i in range(cutoff_time):
        tmp_switchnow = np.zeros_like(switchmap)
        tmp_switchnow[switchmap == i] = 1
        if struc_elem == True:
            tmp_labeled, num_labels = scipy.ndimage.measurements.label(tmp_switchnow, structuring_element)
        elif struc_elem == False:
            tmp_labeled, num_labels = scipy.ndimage.measurements.label(tmp_switchnow)

        tmp_labeled_zoomed = np.kron(tmp_labeled != 0, np.ones((2, 2)))
        contourstest = find_contours(tmp_labeled_zoomed, 0.5)
        newconts = [np.ceil(x / 2.) - 0.5 for x in contourstest]
        contours.append(newconts)
    return contours


def notnan_map(img):
    where_nan = np.isnan(img)
    not_nan = np.logical_not(where_nan)
    return notnan


def notnan_map_series(imgs):
    not_nan_maps = []
    for i in imgs:
        map_ = notnan_map(i)
        not_nan_maps.append(map_)
    return not_nan_maps







def checkequal(data):
    """Checks if all elements in array are equal."""
    return len(set(data)) == 1


def wherechanged(arr):
    where = []
    for i in range(len(arr) - 1):
        diff = arr[i + 1] - arr[i]
        if diff != 0:
            where.append(i + 1)
    return where


def filter_data(data, filtervalue):
    tmp_sizes = []
    tmp_num = []
    for i in range(len(data[1])):
        if testu[1][i] != filtervalue:
            tmp_sizes.append(data[0][i])
            tmp_num.append(data[1][i])
    data[0] = tmp_sizes
    data[1] = tmp_num
    return data




