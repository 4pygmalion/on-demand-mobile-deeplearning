#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import pandas as pd
import numpy as np
sys.path.append('/home/hoheon/packages/')
from utils.serialization import load_pickle
from runtime import GraphPartitioner, read_txt

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('bandwidth', type=int, help='bandwidth(/kbps)')
parser.add_argument('direction', type=str, help='communication direction')
args = parser.parse_args()
bandwidth = args.bandwidth
direction = args.direction

# Load data
server_runtime = pd.read_csv('./result/server_runtime.csv')
client_runtime = pd.read_csv('./result/device_runtime.csv')
d_size = pd.read_csv('./result/data_size.csv')
layer_names = load_pickle('./result/layer_seq.pickle')
client_lib_runtimes = read_txt('./result/runtimes_client.txt')
server_lib_runtimes = read_txt('./result/runtimes_server.txt')


# Prep
layer_mapping = {'input_1':'input', 'dense_1':'dense', 'flatten_1':'flatten', 'input_2':'input'}
layer_names = [layer_mapping[element] if element in layer_mapping.keys() else element for element in layer_names]
server_runtime = server_runtime.replace(layer_mapping)
client_runtime = client_runtime.replace(layer_mapping)
avg_client_runtime = client_runtime.groupby('layer_name').mean().reindex(layer_names)
avg_server_runtime = server_runtime.groupby('layer_name').mean().reindex(layer_names)

# main
def main():
    gp = GraphPartitioner(avg_server_runtime, server_lib_runtimes, avg_client_runtime, client_lib_runtimes, d_size)
    exit_point = gp.get_exitpoint(bandwidth, direction=direction)
    print(exit_point)
    
if __name__ == '__main__':
    main()