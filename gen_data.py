# coding=utf-8
import gym
import gym_duckietown
# from stable_baselines.common.policies import CnnPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
# import time
from numpy import random
import numpy as np
# import matplotlib.pyplot as plt


"""
Description:
    Generate data set with TOT_SAMPLES samples consisting of a sequence of HISTORY_CNT + 1 consecutive observations,
    and additional information, including actions, reward and done indicator.
LIN
2020/03/19
"""

# Image Size CONSTANT
HEIGHT = 480
WIDTH = 640
# Sample Count CONSTANT
TOT_SAMPLES = 5
# History Size CONSTANT
HISTORY_CNT = 4


def rgb2gray(rgb):
    """
    :param rgb: numpy with shape(HEIGHT, WIDTH, 3)
    :return: numpy with shape(HEIGHT, WIDTH)
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def packHistoryVec2CubeHistory(history_vec, action, reward, done, speed):
    """
    pack all info into a sample
    :param history_vec: numpy array with shape (HISTORY_CNT + 1, HEIGHT, WIDTH, 1)
    :param action: numpy array with shape (2,)
    :param reward: float
    :param done: boolean
    :param speed: float (non-negative)
    :return: packed images with shape (HEIGHT + 1, WIDTH, HISTORY_CNT + 1)
    """
    cubeHistory = history_vector[0].reshape((HEIGHT, WIDTH, 1))
    for i in range(HISTORY_CNT):
        cubeHistory = np.append(cubeHistory, history_vector[i+1].reshape((HEIGHT, WIDTH, 1)), axis=2)
    if done:
        done_int = 1
    else:
        done_int = -1
    # TODO: can this following line of code be rewritten in a more decent way?
    tmp_line = np.append(np.append(np.append(np.append(action, reward), done_int), speed), np.zeros((WIDTH - 5,))) \
        .reshape((1, WIDTH, 1))
    tmp_slice = tmp_line
    for i in range(HISTORY_CNT):
        tmp_slice = np.append(tmp_slice, tmp_line, axis=2)
    cubeHistory = np.append(cubeHistory, tmp_slice, axis=0)
    # print(cubeHistory.shape)
    return cubeHistory

def reviseObs(obs):
    """
    return an revised observation
    :return: a gray image representing the observation,
    while the structure is particularly revised for constructing history_vector
    numpy array with shape (1, HEIGHT, WIDTH, 1)
    """
    gray_img = rgb2gray(obs)
    gray_img_with_depth = gray_img.reshape((HEIGHT, WIDTH, 1))
    gray_img_with_depth_and_pre_dimension = gray_img_with_depth[np.newaxis, :]
    return gray_img_with_depth_and_pre_dimension


# Registration of environment
# we need the speed of agent, so turn full_transparency on
env = gym.make('Duckietown-straight_road-v0', full_transparency=True)

steps = 0
done = True
history_vector = []
while steps < TOT_SAMPLES:
    print('step %s' % steps)
    if done:
        # reset the environment and history_vector
        print('one episode done')
        obs = env.reset()
        gray_img = reviseObs(obs)
        history_vector = gray_img
        for i in range(HISTORY_CNT):
            history_vector = np.append(history_vector, gray_img, axis=0)
            # in total, history_vector has HISTORY_CNT + 1 gray images, consisting of almost all elements
            # for a single cubeHistory, except for the substituted image generated via the coming action
    # random.normal scale is the standard deviation, about 95% lies within two-sigma interval
    # a cut off will happen at +/- 1
    # Action = (speed, turning)
    action = np.clip([random.normal(loc=0.6, scale=0.5, size=1),
                      random.normal(loc=0.0, scale=0.3, size=1)], -1, 1)
    # one step ahead, but the validation remains checked soon
    # while even if the validation (i.e. done variable) matters,
    # we still keep record all the situation, for the sake of having abundant samples
    # covering more states
    obs, reward, done, info = env.step(action)
    steps += 1
    gray_img = reviseObs(obs)
    # update the history_vector, which should hold exactly HISTORY_CNT + 1 gray images
    if history_vector.shape[0] == HISTORY_CNT:
        history_vector = np.append(history_vector, gray_img, axis=0)
    else:
        assert(history_vector.shape[0] == HISTORY_CNT + 1)
        for i in range(HISTORY_CNT):
            history_vector[i] = history_vector[i+1]
        history_vector[HISTORY_CNT] = gray_img
    # notice that the agent speed is stacked in the info map, which is a float variable
    cubeHistory = packHistoryVec2CubeHistory(history_vector, action, reward, done, info['Simulator']['robot_speed'])
    # append a sample (cubeHistory) into the whole data set
    # data set will finally be a numpy array with shape (TOT_SAMPLES, HEIGHT + 1, WIDTH, HISTORY_CNT + 1)
    try:
        data
    except NameError:
        data = cubeHistory[np.newaxis, :]
    else:
        data = np.append(data, cubeHistory[np.newaxis, :], axis=0)


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(data[0, :, :, 0], cmap=plt.cm.gray)
# plt.show()

print(data.shape)

np.savez('dataset_vae.npz', data)
