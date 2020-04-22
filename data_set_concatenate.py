import glob
import numpy as np
import argparse
import os

"""
    Last modified on 04/22
    related to v3.1
"""

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--samples', help='samples per small dataset file', type=int, default=300)
args = parser.parse_args()
single_sample_num = args.samples

bigdatafilepath = 'big_data'
if not os.path.exists(bigdatafilepath):
    os.mkdir(bigdatafilepath)
datafilepath = 'data'

os.chdir(datafilepath)
datafiles = glob.glob('dataset_vae_*.bin')
print(len(datafiles))
os.chdir('..')
big_data_set = np.memmap(os.path.join(bigdatafilepath, 'dataset_vae.bin'),
                         dtype='float32', mode='w+', shape=(single_sample_num * len(datafiles), 121, 160, 5))
# input('Enter...')
for i, singleFile in enumerate(datafiles):
    _path = os.path.join(datafilepath, singleFile)
    # single_dataset = np.load(_path)
    # print(single_dataset[0, 0, 0, 0])
    # del single_dataset
    single_dataset = np.memmap(_path, dtype='float32', mode='r', shape=(single_sample_num, 121, 160, 5))
    # single_dataset = single_dataset['arr_0']
    # big_data_set.append(single_dataset)
    # print(single_dataset[0, 0, 0, 0])
    # print('{}-th single Enter...'.format(i))
    # input()
    big_data_set[i*single_sample_num: (i+1)*single_sample_num, :, :, :] = single_dataset
    # print('after {} Enter'.format(i))
    # input()

# big_data_set = np.concatenate(big_data_set)

# print(big_data_set)
# print(big_data_set.shape)
# input('press Enter to continue...')
# np.savez(os.path.join(bigdatafilepath, 'dataset_vae.npz'),
#          big_data_set)
