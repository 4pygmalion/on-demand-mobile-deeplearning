#!/usr/bin/env python
# coding: utf-8


# run time
import timeit

start_time = timeit.default_timer()

from tensorflow import device
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np

with device('CPU:0'):

    sub_model = VGG19(include_top=False, input_shape=(32, 32, 3), classes=10)
    flat = Flatten()(sub_model.layers[-1].output)
    classify_ = Dense(10, activation='softmax')(flat)
    model = Model(inputs=sub_model.inputs, outputs=classify_)
    model.load_weights('./cache/best_param.hdf5')

end_time = timeit.default_timer()
runtime = end_time - start_time

print(runtime)

