# coding=utf-8
__version__ = 'v2'

import gym
import gym_duckietown
from time import time
from numpy import random
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import logging


"""
Description:
    Generate data set with TOT_SAMPLES samples consisting of a sequence of HISTORY_CNT + 1 consecutive observations,
    and additional information, including actions, reward and done indicator.
    Observation consists of images and speed.
    SOLVED: Increase the efficiency, replace numpy.append with built-in list manipulation
LIN
2020/04/01
"""

# define logger (basic config fails because of root handler exists)
logger_file_name = 'log_gen_data.log'
logger = logging.getLogger('gen_data%s' % __version__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler(filename=logger_file_name, mode='a')  # FileHandler returns None
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

logger.info('gen_data-v2')

# Image Size CONSTANT
SCALING_CONSTANT = 1. / 4.
HEIGHT = round(480 * SCALING_CONSTANT)
WIDTH = round(640 * SCALING_CONSTANT)
# Sample Count CONSTANT
TOT_SAMPLES = 300
# History Size CONSTANT
HISTORY_CNT = 4


def rgb2gray(rgb):
    """
    :param rgb: numpy with shape(HEIGHT, WIDTH, 3)
    :return: numpy with shape(HEIGHT, WIDTH)
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def packHistoryVec2CubeHistory(history_vector, speed_vec, action, reward, done):
    """
    pack all info into a sample
    :param history_vector: numpy array with shape (HISTORY_CNT + 1, HEIGHT, WIDTH, 1)
    :param speed_vec: numpy array with shape (HISTORY_CNT + 1,) (non-negative)
    :param action: float
    :param reward: float
    :param done: boolean
    :return: packed images with shape (HEIGHT + 1, WIDTH, HISTORY_CNT + 1)
    """
    cubeHistory = history_vector[0].reshape((HEIGHT, WIDTH, 1))
    for i in range(HISTORY_CNT):
        cubeHistory = np.append(cubeHistory, history_vector[i+1].reshape((HEIGHT, WIDTH, 1)), axis=2)
    if done:
        done_int = 1
    else:
        done_int = 0
    tmp_line = speed_vec
    tmp_line = np.append(tmp_line, [action, reward, done_int])
    tmp_line = np.concatenate((tmp_line, np.zeros((WIDTH - 8,))))
    tmp_line = tmp_line.reshape((1, WIDTH, 1))
    tmp_slice = tmp_line
    for i in range(HISTORY_CNT):
        tmp_slice = np.append(tmp_slice, tmp_line, axis=2)
    cubeHistory = np.append(cubeHistory, tmp_slice, axis=0)
    # print(cubeHistory.shape)
    return cubeHistory

def reviseObs(obs):
    """
    return an revised observation, including modifying the size if necessary
    :return: a gray image representing the observation,
    while the structure is particularly revised for constructing history_vector
    numpy array with shape (1, HEIGHT, WIDTH, 1)
    """
    gray_img = rgb2gray(obs)
    scaled_img = transform.rescale(gray_img, SCALING_CONSTANT, anti_aliasing=True)
    gray_img_with_depth = scaled_img.reshape((HEIGHT, WIDTH, 1))
    gray_img_with_depth_and_pre_dimension = gray_img_with_depth[np.newaxis, :]
    return gray_img_with_depth_and_pre_dimension


def get_action(steps, cur_step):
    cnt_types = 6
    if steps < TOT_SAMPLES / cnt_types:
        return np.clip(random.normal(loc=0.0, scale=0.5, size=1), -1, 1)
    if steps < TOT_SAMPLES * 2 / cnt_types:
        return random.uniform(low=-1.0, high=1.0, size=1)
    if steps < TOT_SAMPLES * 4 / cnt_types:
        if cur_step < 4:
            return 0.0
        if steps < TOT_SAMPLES * 3 / cnt_types:
            return -1.0
        return 1.0
    if steps < TOT_SAMPLES * 5 /cnt_types:
        if cur_step % 2 == 0:
            return 1.0
        else:
            return -1.0
    return 0.0


# Registration of environment
# we need the speed of agent, so turn full_transparency on
env = gym.make('Duckietown-loop_obstacles-v0', full_transparency=True)

# variables for time calculation
sum_time = [0, 0, 0, 0]
start_time = time()

steps = 0
real_steps = 0
done = True
history_vector = []
speed_vector = []
data = []
cnt_done = 0
cur_step = 0
while steps < TOT_SAMPLES:
    # print('step %s' % steps)
    real_steps += 1
    cur_step += 1
    if done:
        cur_step = 0
        # reset the environment and history_vector
        print('one episode done')
        obs = env.reset()
        gray_img = reviseObs(obs)
        history_vector = gray_img
        for i in range(HISTORY_CNT - 1):
            history_vector = np.append(history_vector, gray_img, axis=0)
            # here, history_vector has HISTORY_CNT gray images, and one more will be appended later
            # in total, history_vector has HISTORY_CNT + 1 gray images, consisting of almost all elements
            # for a single cubeHistory, except for the substituted image generated via the coming action
        speed_vector = np.array([0.0 for i in range(HISTORY_CNT)])
    # random.normal scale is the standard deviation, about 95% lies within two-sigma interval
    # a cut off will happen at +/- 1
    # Action = (1.0, steering)
    action = get_action(steps, cur_step)
    # one step ahead, but the validation remains checked soon
    # while even if the validation (i.e. done variable) matters,
    # we still keep record all the situation, for the sake of having abundant samples
    # covering more states
    time_0 = time()
    obs, reward, done, info = env.step([1.0, action])
    sum_time[0] += time() - time_0

    steps += 1
    predicted_speed = info['Simulator']['robot_speed']
    gray_img = reviseObs(obs)
    # update the history_vector, which should hold exactly HISTORY_CNT + 1 gray images
    # further assume that shape of history_vector equals to shape of speed_vector
    assert(history_vector.shape[0] == speed_vector.shape[0])
    time_0 = time()
    if history_vector.shape[0] == HISTORY_CNT:
        history_vector = np.append(history_vector, gray_img, axis=0)
        speed_vector = np.append(speed_vector, predicted_speed)
    else:
        assert(history_vector.shape[0] == HISTORY_CNT + 1)
        for i in range(HISTORY_CNT):
            history_vector[i] = history_vector[i+1]
            speed_vector[i] = speed_vector[i+1]
        history_vector[HISTORY_CNT] = gray_img
        speed_vector[HISTORY_CNT] = predicted_speed
    sum_time[1] += time() - time_0
    # neglect samples where there exists at least two same images
    if cur_step < 4:
        steps -= 1
        continue
    # the history will be truly counted into the data set here, after all previous validation
    if done:  # auxiliary statistics for counting episodes number and its length
        cnt_done += 1
    # notice that the agent speed is stacked in the info map, which is a float variable
    time_0 = time()
    cubeHistory = packHistoryVec2CubeHistory(history_vector, speed_vector, action, reward, done)
    sum_time[2] += time() - time_0
    # append a sample (cubeHistory) into the whole data set
    # data set will finally be a numpy array with shape (TOT_SAMPLES, HEIGHT + 1, WIDTH, HISTORY_CNT + 1)
    # variable data is the built-in list
    time_0 = time()
    data.append(cubeHistory.tolist())
    sum_time[3] += time() - time_0
    print('after step %s' % steps)

data = np.array(data, dtype=float)

done_id = 4
fig = plt.figure()
for i in range(0, 100):
    index = i
    ax = fig.add_subplot(10, 10, i+1)
    ax.axis('off')
    ax.imshow(data[index, :HEIGHT, :, 0], cmap=plt.cm.gray)
    done_int = data[index, HEIGHT, 7, 0]
    ax.set_title('done: %s' % done_int, size=5, fontweight='normal', pad=2.0)
    if done_int == 1.0:
        done_id = index
plt.subplots_adjust(hspace=0.4, wspace=0.1)
plt.savefig('sampleDataset.svg', format='svg', dpi=1200)

fig = plt.figure()
for i in range(0, 5):
    for j in range(-1, 2):
        ax = fig.add_subplot(3, 5, (j+1)*5+i+1)
        ax.axis('off')
        ax.imshow(data[done_id + j, :HEIGHT, :, i], cmap=plt.cm.gray)
plt.subplots_adjust(hspace=0.1)
plt.savefig('sampleHistoryImg.svg', format='svg', dpi=1200)

logger.info('data shape is ' + str(data.shape))
logger.info('total episodes %s' % cnt_done)
logger.info('average steps per episode: %s' % (TOT_SAMPLES / cnt_done))
logger.info('----')
logger.info('average time for part 1: ' + str(sum_time[0] / real_steps) + 'sec')
logger.info('average time for part 2: ' + str(sum_time[1] / real_steps) + 'sec')
logger.info('average time for part 3: ' + str(sum_time[2] / steps) + 'sec')
logger.info('average time for part 4: ' + str(sum_time[3] / steps) + 'sec')
print('info are saved into log file and image files')

np.savez('dataset_vae.npz', data)

print('completed: data set generation')
logger.info('time used: ' + str(time() - start_time) + 'sec')
