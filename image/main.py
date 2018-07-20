
from skimage import io
import numpy as np

def load_data_nchw_float(filename):
    f = open(filename)
    line = f.readline()
    n, c, h, w = line.split(' ')
    n, c, h, w = int(n), int(c), int(h), int(w)
    print("n = %d, c = %d, h = %d, w = %d" % (n, c, h, w))
    line = f.readline()
    tmp = line.split(' ')
    if (tmp[-1] == ''):
        tmp = tmp[:-1]
    data = list(map(float, tmp))
    if (len(data) == n*c*h*w):
        data = np.array(data).reshape(c, h, w).transpose(1, 2, 0)
        print(data.shape)
        return data
    else:
        print("n*c*h*w = %d, len(data) = %d" % (n*c*h*w, len(data)))
        exit()

def load_data_nchw_int(filename):
    f = open(filename)
    line = f.readline()
    n, c, h, w = line.split(' ')
    n, c, h, w = int(n), int(c), int(h), int(w)
    print("n = %d, c = %d, h = %d, w = %d" % (n, c, h, w))
    line = f.readline()
    tmp = line.split(' ')
    if (tmp[-1] == ''):
        tmp = tmp[:-1]
    data = list(map(np.uint8, tmp))
    if (len(data) == n*c*h*w):
        data = np.array(data).reshape(c, h, w).transpose(1, 2, 0)
        print(data.shape)
        return data
    else:
        print("n*c*h*w = %d, len(data) = %d" % (n*c*h*w, len(data)))
        exit()

def main():
    io.imsave("1.jpg", load_data_nchw_float("1.txt"))
    # io.imsave("db.jpg", load_data_nchw_int("db.txt"))
    # io.imsave("input2.jpg", load_data_nchw_int("input2.txt"))

if __name__ == '__main__':
    main()
