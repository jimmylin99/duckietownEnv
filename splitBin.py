import numpy as np

file = np.memmap('big_data/dataset_vae.bin', shape=(30000, 121, 160, 5),
                 mode='r', dtype='float32', offset=)

import numpy.lib.format
