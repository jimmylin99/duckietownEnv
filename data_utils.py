import numpy as np
from skimage import transform

"""
    Last modified on 04/22
    related to v3.1: rescale image to [0, 1]
"""

# History Size CONSTANT
HISTORY_CNT = 4
# Image Size CONSTANT
SCALING_CONSTANT = 1. / 4.
HEIGHT = round(480 * SCALING_CONSTANT)
WIDTH = round(640 * SCALING_CONSTANT)


def reviseObs(obs):
    """
    return an revised observation, including modifying the size if necessary
    :param obs: pure observation got by calling env.reset() or env.step()
    :return: a gray image representing the observation,
    while the structure is particularly revised for constructing history_vector
    numpy array with shape (1, HEIGHT, WIDTH, 1)
    """

    def rgb2gray(rgb):
        """
        :param rgb: numpy with shape(HEIGHT, WIDTH, 3)
        :return: numpy with shape(HEIGHT, WIDTH)
        """
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    gray_img = rgb2gray(obs)
    scaled_img = transform.rescale(gray_img, SCALING_CONSTANT, anti_aliasing=True)
    # new added
    scaled_img /= 255
    gray_img_with_depth = scaled_img.reshape((HEIGHT, WIDTH, 1))
    gray_img_with_depth_and_pre_dimension = gray_img_with_depth[np.newaxis, :]
    return gray_img_with_depth_and_pre_dimension


def pack4input(history_vector, steering):
    """
    pack all info into a sample
    :param history_vector: numpy array with shape (HISTORY_CNT, HEIGHT, WIDTH, 1)
    :param steering: float in range [-1.0, 1.0]
    # detail for last row
    # placeholder speed_vec: numpy array with shape (HISTORY_CNT + 1,) (non-negative)
    # steering: located at [HEIGHT, HISTORY + 1, 1]
    # placeholders
    :return: packed images with shape (HEIGHT + 1, WIDTH, HISTORY_CNT + 1)
    """
    cubeHistory = np.zeros((HEIGHT, WIDTH, 1))
    for i in range(HISTORY_CNT):
        cubeHistory = np.append(cubeHistory, history_vector[i].reshape((HEIGHT, WIDTH, 1)), axis=2)

    tmp_line = np.zeros((HISTORY_CNT + 1,))
    tmp_line = np.append(tmp_line, steering)
    tmp_line = np.concatenate((tmp_line, np.zeros((WIDTH - HISTORY_CNT - 2,))))
    tmp_line = tmp_line.reshape((1, WIDTH, 1))
    tmp_slice = tmp_line
    for i in range(HISTORY_CNT):
        tmp_slice = np.append(tmp_slice, tmp_line, axis=2)
    cubeHistory = np.append(cubeHistory, tmp_slice, axis=0)
    # print(cubeHistory.shape)
    return cubeHistory
