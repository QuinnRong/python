#!/usr/bin/python3

from caffe.proto import caffe_pb2
from google.protobuf import text_format

import caffe
import numpy as np

def get_layers_by_name(prototxt, name):
    layers = []

    proto = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(), proto)
    for i in range(len(proto.layer)):
        layer = proto.layer[i]
        if layer.type == name:
            layers.append(layer.name)

    return layers

def set_random_weights(prototxt, layers):
    net = caffe.Net(prototxt, caffe.TEST)

    for layer_name in layers:
      for i in range(len(net.params[layer_name])):
          net.params[layer_name][i].data[...] = np.random.random_sample(net.params[layer_name][i].data.shape)
          print(layer_name, ": ", net.params[layer_name][i].data.shape)

    net.save("model.caffemodel")

def main():
    layers = get_layers_by_name("model.prototxt", "Convolution")
    set_random_weights("model.prototxt", layers)

if __name__ == '__main__':
    main()
