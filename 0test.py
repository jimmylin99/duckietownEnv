import numpy as np

b = np.array([[2, 3, 4], [1, 0 ,8]]).reshape((2, 3, 1))


def add_img():
    try:
        global arr
        arr
    except NameError:
        # global arr
        print('Name Error')
        arr = b
    else:
        # global arr
        arr = np.append(arr, b, axis=2)


add_img()
add_img()
del arr
add_img()
print(arr)
print(arr.shape)

