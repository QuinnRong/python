#! /usr/bin/env python3
import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def load_feature(fileName, data):
    '''
    input: txt file
    format:
    [0,0,0] val
    [1,0,0] val
    output: a list of values
    '''
    with open(fileName) as f:
        lines = f.readlines() 
        for line in lines:
            lineData=line.split(' ')
            x = float(lineData[-1])
            data.append(x)

def cal_cos_distance(data1, data2):
    '''
    input: two lists
    output: cos distance = data1 dot data2 / norm1 / norm2
    '''
    norm1 = np.sqrt(np.dot(data1, data1))
    print("norm1:")
    print(norm1)
    norm2 = np.sqrt(np.dot(data2, data2))
    print("norm2:")
    print(norm2)
    return np.dot(data1, data2) / norm1 / norm2

def cos_distance(file1, file2):
    '''
    input: two files
    format:
    [0,0,0] val
    [1,0,0] val
    output: cos distance
    '''
    data_1 = []
    load_feature(file1, data_1)
    data_2 = []
    load_feature(file2, data_2)
    print("cos diatance is:", cal_cos_distance(data_1, data_2))

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
        return data
    else:
        print("n*c*h*w = %d, len(data) = %d" % (n*c*h*w, len(data)))
        exit()

def run_tf(sess, input_x, output_y, input_file, output_file, invert, normalize):
    '''
    nchw => nhwc (=> invert to RGB) (=> normalize)
    output format:
    [0,0,0] val
    [1,0,0] val
    '''
    data = []
    print("input:", input_file)
    data.append(load_txt_nchw("input/" + input_file))
    # convert to tensorflow format:
    # nchw to nhwc
    data = np.array(data).transpose(0, 2, 3, 1).astype(np.float32)
    if invert:
        print("RGB to BGR...")
        [n, h, w, c] = data.shape
        data = np.append(np.append(data[:,:,:,2], data[:,:,:,1]), data[:,:,:,0]).reshape(n, c, h, w).transpose(0, 2, 3, 1)
        print(data.shape)
    if normalize:
        print("normalize...")
        data = data / 255.0

    output = sess.run(output_y.name, feed_dict={input_x.name:data})
    print("output shape:", output.shape)
    print(output)

    print("output:", output_file)
    f = open("output/" + output_file, 'w')
    output = output.reshape((-1,))
    for idx in range(len(output)):
        f.write("[%d,%d,%d] %s" % (idx, 0, 0, str(output[idx])+"\n"))
    f.close()
    print()

def run(input_list, output_list, model_file, invert=False, normalize=False):
    '''
    input_file: txt data(int)
    n c h w
    val val...
    output_file format:
    [0,0,0] -134.75
    [1,0,0] -79.75
    model_file: folder containing meta, data, index
    obtained by saver.save(sess, model_file)
    '''
    if not os.path.exists("output"):
        os.mkdir("output")

    print("run tensorflow...")
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_file + ".meta")
        saver.restore(sess, model_file)
        # get op name
        ops = [o for o in sess.graph.get_operations()]
        f = open("output/op.txt", "w")
        for o in ops:
           print(o.name, file=f)
        f.close()
        # get global var name
        # f = open("output/gv.txt", "w")
        # gv = [v for v in tf.global_variables()]
        # for v in gv:
        #    print(v.name, file=f)
        # f.close()
        graph = tf.get_default_graph()
        input_op = ops[0]
        input_x = graph.get_operation_by_name(input_op.name).outputs[0]
        print("input tensor name:", input_x.name)
        for idx in range(1, len(ops)):
            if "init" not in ops[-idx].name and "save" not in ops[-idx].name:
                output_op = ops[-idx]
                break
        output_y = graph.get_operation_by_name(output_op.name).outputs[0]
        print("output tensor name:", output_y.name)

        for i in range(len(input_list)):
            run_tf(sess, input_x, output_y, input_list[i], output_list[i], invert, normalize)

def run_10_files():
    input_list = []
    output_list = []
    for i in range(1, 11):
        input_list.append(str(i)+".txt")
        output_list.append("output_"+str(i)+".txt")
    run(input_list, output_list, "./movidius_model/extract_inference", True, True)

def main():
    run_10_files()

if __name__ == '__main__':
    sys.exit(main())
