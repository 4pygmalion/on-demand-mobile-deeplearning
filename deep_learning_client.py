#!/usr/bin/env python
# coding: utf-8

# Load library
import sys
import argparse
import numpy as np

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow import device

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('exitpoint', type=int, help='outbound exitpoint')
args = parser.parse_args()
exitpoint = args.exitpoint


# Given data
x_test = np.load('./data/x_test.npy') 

with device('CPU'):
    # Build model
    sub_model = VGG19(include_top=False, input_shape=(32, 32, 3), classes=10)
    flat = Flatten()(sub_model.layers[-1].output)
    classify_ = Dense(10, activation='softmax')(flat)
    model = Model(inputs=sub_model.inputs, outputs=classify_)
    model.load_weights('./cache/best_param.hdf5')
    model = Model(model.layers[0].input, model.layers[exitpoint].output)
    
    
    result = model(x_test)
    np.save('/home/hoheon/Jupyter/2021/EdgeCom/data/mid_data', result.numpy())