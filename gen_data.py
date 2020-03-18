import gym
import gym_duckietown
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import time
from numpy import random
import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def incImgDim(tmp_img):

    return


def packImg2CubeHistory(new_img):
    """
    acts on global variable trans_imgs
    pack single series of images, converting several (maybe 5) images from 2 dimensions to 3 dimensions
    :param new_img: numpy array with shape (height, width)
    :return: packed images with shape (height, width, depth)
    """
    revised_img = incImgDim(new_img)
    try:
        global trans_imgs
        trans_imgs
    except NameError:
        # global arr
        print('Name Error')
        trans_imgs = revised_img
    else:
        # global arr
        trans_imgs = np.append(trans_imgs, revised_img, axis=2)


env = gym.make('Duckietown-straight_road-v0')

obs = env.reset()
grayImg = rgb2gray(obs)
imgs = np.array(grayImg)[np.newaxis, :]

steps = 0
while steps < 500:
    print(steps)
    obs = env.reset()
    grayImgSet = rgb2gray(obs)
    actionStep = 0
    Action = (random.normal(loc=0.8, scale=0.2, size=1), random.uniform(-1, 1, size=1))
    done = True
    while actionStep < 5 - 1:
        obs, reward, done, ___ = env.step(Action)

        grayImg = rgb2gray(obs)
        # _grayImg = np.array(grayImg)[np.newaxis, :]
        grayImgSet = np.append(grayImgSet, grayImg, axis=0)
        actionStep = actionStep + 1
        if done:
            break
    # Action = (random.normal(loc=0.8, scale=, size=1), random.uniform(-1, 1, size=1))
    # obs, _, done, ___ = env.step(Action)
    # if done:
    #     obs = env.reset()
    # imgs = np.array(imgs)[np.newaxis, :]
    # _grayImg = np.array(_grayImg)[np.newaxis, :]
    if done:
        obs = env.reset()
        continue
    _actionImg = np.append(np.array(Action), np.zeros(grayImg.shape[1] - 2))
    # print(_actionImg)
    # print(_actionImg.shape)
    # time.sleep(10)
    grayImgSet = np.append(grayImgSet, np.reshape(_actionImg, [1, 640]), axis=0)
    _grayImgSet = np.array(grayImgSet)[np.newaxis, :]
    if steps == 0:
        imgs = _grayImgSet
    else:
        imgs = np.append(imgs, _grayImgSet, axis=0)
    steps = steps + 1
    # print(obs.shape)
    # grayImg = rgb2gray(obs)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(imgs[0], cmap=plt.cm.gray)
# ax = fig.add_subplot(222)
# ax.imshow(obs)
# ax = fig.add_subplot(223)
# ax.imshow(grayImg)
# ax = fig.add_subplot(224)
# ax.imshow(obs, cmap=plt.cm.gray)
plt.show()

print(grayImgSet.shape)
print(imgs.shape)

np.savez('duckie_img.npz', imgs)
