from skimage import io, transform
import numpy as np
import os
import math
import time

def save_txt_nchw(filename, data, N, C, H, W):
    '''
    input: 3d array((c h w))
    output: txt data with the same dtype as input
    n c h w
    val val...
    '''
    print("------------")
    print("input size : n=%d, c=%d, h=%d, w=%d" % (N, C, H, W))
    dtype = type(data[0][0][0])
    print("data type  :", dtype)
    print("output size:", data.shape)
    print("------------")
    f = open(filename, 'w')
    print("%d %d %d %d" % (N, C, H, W), file = f)
    for c in range(C):
        for h in range(H):
            for w in range(W):
                f.write(str(data[h, w, c]) + " ")
    f.close()

def load_txt_nchw(filename):
    '''
    input: txt data(int or float)
    n c h w
    val val...
    output: 3d array((c h w)) with the same dtype as input
    '''
    f = open(filename)
    line = f.readline()
    n, c, h, w = line.split(' ')
    n, c, h, w = int(n), int(c), int(h), int(w)
    print("------------")
    print("input size : n=%d, c=%d, h=%d, w=%d" % (n, c, h, w))

    line = f.readline()
    tmp = line.split(' ')
    if (tmp[-1] == ''):
        tmp = tmp[:-1]
    dtype = type(eval(tmp[0]))
    print("data type  :", dtype)
    if dtype == int:
        data = list(map(np.uint8, tmp))
    elif dtype == float:
        data = list(map(float, tmp))
    else:
        print("data type not recognized: %s", type(eval(tmp[0])))

    if (len(data) == n*c*h*w):
        data = np.array(data).reshape(c, h, w)
        print("output size:", data.shape)
        print("------------")
        return data
    else:
        print("n*c*h*w = %d, len(data) = %d" % (n*c*h*w, len(data)))
        exit()

def txt2img(name):
    '''
    input: txt data(uint8 or float)
    n c h w
    val val...
    output: an image
    '''
    data = load_txt_nchw("data/"+name+".txt")
    data = data.transpose(1, 2, 0)
    io.imsave("data/"+name+".jpg", data)

def img2txt(name):
    '''
    input: an image
    output: txt data(uint8)
    n c h w
    uint8 uint8...
    '''
    data = io.imread("data/"+name+".jpg")
    [h, w, c] = data.shape
    save_txt_nchw("data/"+name+".txt", data, 1, c, h, w)

def shuffle_rgb(filename):
    '''
    input: an image
    output: a folder named RGB containing 12 images with different colors
    '''
    if not os.path.exists("RGB"):
        os.mkdir("RGB")

    data = io.imread("data/"+filename)
    [h, w, c] = data.shape
    print("------------")
    print("input size :", data.shape)
    print("------------")
    zero = np.zeros(data[:,:,0].size, dtype=np.uint8)

    io.imsave("RGB/R.jpg", data[:,:,0])
    io.imsave("RGB/G.jpg", data[:,:,1])
    io.imsave("RGB/B.jpg", data[:,:,2])

    new_data = np.append(np.append(data[:,:,0], zero), zero).reshape(c, h, w).transpose(1, 2, 0)
    io.imsave("RGB/R00.jpg", new_data)
    new_data = np.append(np.append(zero, data[:,:,1]), zero).reshape(c, h, w).transpose(1, 2, 0)
    io.imsave("RGB/0G0.jpg", new_data)
    new_data = np.append(np.append(zero, zero), data[:,:,2]).reshape(c, h, w).transpose(1, 2, 0)
    io.imsave("RGB/00B.jpg", new_data)

    new_data = np.append(np.append(data[:,:,0], data[:,:,1]), data[:,:,2]).reshape(c, h, w).transpose(1, 2, 0)
    io.imsave("RGB/RGB.jpg", new_data)
    new_data = np.append(np.append(data[:,:,2], data[:,:,1]), data[:,:,0]).reshape(c, h, w).transpose(1, 2, 0)
    io.imsave("RGB/BGR.jpg", new_data)
    new_data = np.append(np.append(data[:,:,1], data[:,:,2]), data[:,:,0]).reshape(c, h, w).transpose(1, 2, 0)
    io.imsave("RGB/GBR.jpg", new_data)
    new_data = np.append(np.append(data[:,:,0], data[:,:,2]), data[:,:,1]).reshape(c, h, w).transpose(1, 2, 0)
    io.imsave("RGB/RBG.jpg", new_data)
    new_data = np.append(np.append(data[:,:,1], data[:,:,0]), data[:,:,2]).reshape(c, h, w).transpose(1, 2, 0)
    io.imsave("RGB/GRB.jpg", new_data)
    new_data = np.append(np.append(data[:,:,2], data[:,:,0]), data[:,:,1]).reshape(c, h, w).transpose(1, 2, 0)
    io.imsave("RGB/BRG.jpg", new_data)

def radian(ang):
    return ang/180*math.pi

def rotate(x, y, ang):
    sin, cos = math.sin(radian(ang)), math.cos(radian(ang))
    return x*cos - y*sin, x*sin + y*cos

def interpolation(x, y, data, mode):
    if mode=="nn":
        x, y = round(x), round(y)
        res = data[y,x,:]
    elif mode=="bilinear":
        x0, x1 = math.floor(x), math.ceil(x)
        y0, y1 = math.floor(y), math.ceil(y)
        py0 = (x-x0)*(data[y0,x1,:] - data[y0,x0,:]) + data[y0,x0,:]
        py1 = (x-x0)*(data[y1,x1,:] - data[y1,x0,:]) + data[y1,x0,:]
        res = (y-y0)*(py1 - py0) + py0
    else:
        print("mode not valid!")
        exit()
    return res

def img_rotate_forward(filename, ang):
    '''
    w => x
    h => y
    ang => anticlock
    '''
    if not os.path.exists("rotate"):
        os.mkdir("rotate")
    sin, cos = math.sin(radian(ang)), math.cos(radian(ang))

    data = io.imread(filename)
    [H, W, c] = data.shape
    W0 = math.ceil(abs(H*sin) + abs(W*cos))
    H0 = math.ceil(abs(H*cos) + abs(W*sin))
    new_data = np.ones((H0, W0, c), dtype=np.uint8)*255
    
    for h in range(H):
        for w in range(W):
            w0 = round((w - W/2)*cos - (h - H/2)*sin + W0/2)
            h0 = round((w - W/2)*sin + (h - H/2)*cos + H0/2)
            new_data[h0,w0,:] = data[h,w,:]

    io.imsave("rotate/"+str(ang)+"-forward.jpg", new_data)

def img_rotate_backward(filename, ang, mode):
    '''
    w => x
    h => y
    ang => anticlock
    '''
    if not os.path.exists("rotate"):
        os.mkdir("rotate")
    sin, cos = math.sin(radian(ang)), math.cos(radian(ang))

    data = io.imread("data/"+filename).astype(float)    # float is importent for interpolation!!!
    [H, W, c] = data.shape
    print("input image size: h=%d, w=%d" % (H, W))
    W0 = math.ceil(abs(H*sin) + abs(W*cos))
    H0 = math.ceil(abs(H*cos) + abs(W*sin))
    print("output image size: h=%d, w=%d" % (H0, W0))
    new_data = np.ones((H0, W0, c), dtype=np.uint8)*255
    
    time_start = time.time()
    for h0 in range(H0):
        for w0 in range(W0):
            w = (w0 - W0/2)*cos + (h0 - H0/2)*sin + W/2
            h = -(w0 - W0/2)*sin + (h0 - H0/2)*cos + H/2
            if w>0 and w<(W - 1) and h>0 and h<(H - 1):
                new_data[h0,w0,:] = interpolation(w, h, data, mode)
    time_end = time.time()
    print("time cost: %.2f" % (time_end - time_start))
    io.imsave("rotate/"+str(ang)+"-backward-"+mode+".jpg", new_data)

def resize(filename, H0, W0, mode):
    '''
    input: an image
    output: a folder named resize containing resized image and txt data(uint8)
    '''
    if not os.path.exists("resize"):
        os.mkdir("resize")
    data = io.imread("data/"+filename)
    [H, W, c] = data.shape
    # new_data = transform.resize(data,(H, W), mode='reflect', preserve_range=True).astype(np.uint8)
    new_data = np.ones((H0, W0, c), dtype=np.uint8)*255

    time_start = time.time()
    for h0 in range(H0):
        for w0 in range(W0):
            h = 1.0*H/H0*h0
            w = 1.0*W/W0*w0
            if w>0 and w<(W - 1) and h>0 and h<(H - 1):
                new_data[h0,w0,:] = interpolation(w, h, data, mode)
    time_end = time.time()
    print("time cost: %.2f" % (time_end - time_start))
    save_txt_nchw("resize/"+str(H0)+"-"+str(W0)+".txt", new_data, 1, c, H0, W0)
    io.imsave("resize/"+str(H0)+"-"+str(W0)+".jpg", new_data)

def invertRB(filename):
    '''
    input: txt data(RGB or BGR)
    n c h w
    val val...
    output: txt data and image(BGR or RGB)
    '''
    if not os.path.exists("invertRB"):
        os.mkdir("invertRB")

    data = load_txt_nchw("data/"+filename)
    data = data.transpose(1, 2, 0)
    [h, w, c] = data.shape
    new_data = np.append(np.append(data[:,:,2], data[:,:,1]), data[:,:,0]).reshape(c, h, w).transpose(1, 2, 0)
    save_txt_nchw("invertRB/origin.txt", data, 1, c, h, w)
    save_txt_nchw("invertRB/invert.txt", new_data, 1, c, h, w)
    io.imsave("invertRB/origin.jpg", data)
    io.imsave("invertRB/invert.jpg", new_data)

def txt_equal(file1, file2):
    '''
    input: two txt data
    n c h w
    val val...
    output: euqal or not
    '''
    data1 = load_txt_nchw(file1)
    data2 = load_txt_nchw(file2)
    [C, H, W] = data1.shape
    for c in range(C):
        for h in range(H):
            for w in range(W):
                if not data1[c,h,w] == data2[c,h,w]:
                    print("%s and %s are not equal." % (file1, file2))
                    idx = c*H*W + h*W + w
                    print("c=%d, h=%d, w=%d" % (c, h, w))
                    print("idx=%f, data1=%f, data2=%f" % (idx, data1[c,h,w], data2[c,h,w]))
                    return
    print("%s and %s are equal." % (file1, file2))

def main():
    img2txt("Lenna")
    txt2img("Lenna")
    resize("Lenna.jpg", 240, 320, "nn")
    invertRB("Lenna.txt")
    shuffle_rgb("Lenna.jpg")
    img_rotate_backward("Lenna.jpg", 30, "nn")
    img_rotate_backward("Lenna.jpg", 40, "bilinear")

if __name__ == '__main__':
    main()
