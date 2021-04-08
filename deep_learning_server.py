#!/usr/bin/env python
# coding: utf-8

# Load library
import sys
import argparse
import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow import device

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('exitpoint', type=int, help='inbound exitpoint')
args = parser.parse_args()
exitpoint = args.exitpoint


# Build model
sub_model = VGG19(include_top=False, input_shape=(32, 32, 3), classes=10)
flat = Flatten()(sub_model.layers[-1].output)
classify_ = Dense(10, activation='softmax')(flat)
model = Model(inputs=sub_model.inputs, outputs=classify_)
model.load_weights('/home/hoheon/Jupyter/2021/EdgeCom/cache/best_param.hdf5')

def main():
    data = np.load('/home/hoheon/Jupyter/2021/EdgeCom/data/mid_data_server.npy')
    subgraph = Sequential()
    for layer in model.layers[exitpoint:]:
        subgraph.add(layer)

    print(subgraph(data))
    
if __name__ == '__main__':
    with device('CPU:0'):
        main()