#! /usr/bin/env python3
import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def loadData(fileName, data):
    '''
    input: txt file
    format:
    [0,0,0] -134.75
    [1,0,0] -79.75
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
    [0,0,0] -134.75
    [1,0,0] -79.75
    output: cos distance
    '''
    data_1 = []
    loadData(file1, data_1)
    data_2 = []
    loadData(file2, data_2)
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

def run_tf(input_file, output_file, model_file):
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

    data = []
    data.append(load_txt_nchw(input_file))
    # convert to tensorflow format(nhwc, 0~1)
    data = np.array(data).transpose(0, 2, 3, 1).astype(np.float32)
    data = data / 255.0
    print("input(float32):", data[0,0,0,:])

    print("run tensorflow...")
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_file + ".meta")
        saver.restore(sess, model_file)
        # get op name
        f = open("output/op.txt", "w")
        ops = [o for o in sess.graph.get_operations()]
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

        output_y = sess.run(output_y.name, feed_dict={input_x.name:data})
        print("output shape:", output_y.shape)
        print(output_y)

        f = open(output_file, 'w')
        output = output_y.reshape((-1,))
        for idx in range(len(output)):
            f.write("[%d,%d,%d] %s" % (idx, 0, 0, str(output[idx])+"\n"))
        f.close()

def main():
    run_tf("input/db_bgr.txt", "output/output_db_bgr_tf.txt", "./movidius_model/extract_inference")
    cos_distance("output/output_db_bgr_tf.txt", "output/output_db_rgb_tf.txt")

if __name__ == '__main__':
    sys.exit(main())
