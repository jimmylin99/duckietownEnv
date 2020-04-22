import numpy as np
from numpy.lib.format import open_memmap
from numpy.lib.format import _filter_header, _wrap_header


def header2str(d):
    header = ["{"]
    for key, value in sorted(d.items()):
        # Need to use repr here, since we eval these when reading
        header.append("'%s': %s, " % (key, repr(value)))
    header.append("}")
    header = "".join(header)
    header = _filter_header(header)
    header = _wrap_header(header, (1, 0))
    return header


filename = 'big_data/dataset_vae.bin'
file = np.memmap(filename, shape=(300000, 121, 160, 5),
                 mode='r', dtype='float32')

print(file[0, 0, 0, 0])

header = np.lib.format.header_data_from_array_1_0(file)
headerStr = header2str(header)
with open(filename, 'r+b') as f:
    content = f.read()
    # f.seek(0, 0)
    # headerStr = header2str(header)
    # f.write(headerStr + content)
with open(filename, 'w+b') as f:
    f.write(headerStr)
with open(filename, 'a+b') as f:
    f.write(content)

# data = open_memmap('big_data/dataset_vae.npy', shape=file.shape, dtype=file.dtype,
#                    mode='w+')
#
# data[:, :, :, :] = file[:, :, :, :]
#
# # np.savez('big_data/dataset_vae.npz', data)
