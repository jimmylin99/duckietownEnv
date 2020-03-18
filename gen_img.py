import gym
import gym_duckietown

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import time

env = gym.make('Duckietown-straight_road-v0')

from numpy import random
import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# while True:
#     obs = env.reset()
#     env.render()
#     time.sleep(2)
obs = env.reset()
grayImg = rgb2gray(obs)
imgs = np.array(grayImg)[np.newaxis, :]
# print(type(imgs), imgs.shape)
# imgs = np.append(imgs, grayImg, axis=0)
# print(type(imgs), imgs.shape)

# a = np.array([[1, 2, 3], [4, 5, 6]])
# print(np.append(a, a, axis=0).shape)
# print(a.shape)

steps = 0
while steps < 5000 - 1:
    # action = (random.normal(loc=0.8, scale=, size=1), random.uniform(-1, 1, size=1))
    # obs, _, done, ___ = env.step(action)
    # if done:
    #     obs = env.reset()
    obs = env.reset()
    grayImg = rgb2gray(obs)
    _grayImg = np.array(grayImg)[np.newaxis, :]
    imgs = np.append(imgs, _grayImg, axis=0)
    steps = steps + 1

    # print(obs.shape)
    # grayImg = rgb2gray(obs)
    # fig = plt.figure()
    # ax = fig.add_subplot(221)
    # ax.imshow(grayImg, cmap=plt.cm.gray)
    # ax = fig.add_subplot(222)
    # ax.imshow(obs)
    # ax = fig.add_subplot(223)
    # ax.imshow(grayImg)
    # ax = fig.add_subplot(224)
    # ax.imshow(obs, cmap=plt.cm.gray)
    # plt.show()


print(imgs.shape)

np.savez('duckie_img.npz', imgs)
