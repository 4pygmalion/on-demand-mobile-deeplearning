import numpy as np
import itertools
import timeit
import tensorflow as tf


class GraphPartitioner(object):

    def __init__(self, server_profile, server_lib_runtime, client_profile, client_lib_runtime, dsize):
        self.server_profile = server_profile
        self.server_lib_runtime = np.mean(server_lib_runtime)
        self.client_profile = client_profile
        self.client_lib_runtime = np.mean(client_lib_runtime)
        self.dsize = dsize
        self.n_layers = len(server_profile)


    def get_exitpoint(self, bandwidth, direction='single'):
        if direction == 'single':
            runtimes = self.search_exitpoint_single(bandwidth)
            return np.argmin(runtimes[1:])
        else:
            runtimes = self.search_exitpoint_bidirection(bandwidth)
            _masked_runtime = runtimes[1:, 1:].copy()
            _masked_runtime[np.where(_masked_runtime == 0)] = 999

            min_runtime = _masked_runtime.min()
            outpoint, inpoint = np.where(_masked_runtime == min_runtime)

            return outpoint[0], inpoint[0]


    def search_exitpoint_single(self, bandwith, batch_size=None, min_latency=None):
        '''
        Parameters
        ----------
        bandwith: /kbps
        '''

        durations = np.zeros(shape=(self.n_layers))

        for i in range(0, self.n_layers):
            if batch_size is None:
                t_dsize = self.dsize['dsize'].iloc[i] * 1
                client_runtime = self.client_profile['runtime'].iloc[0: i].sum()
                server_runtime = self.server_profile['runtime'].iloc[i:].sum()
            else:
                client_runtime = self.client_profile['runtime'].iloc[0: i].sum() * batch_size
                server_runtime = self.server_profile['runtime'].iloc[i:].sum() * batch_size
                t_dsize = self.dsize['dsize'].iloc[i] * batch_size

            trans_latency = t_dsize / bandwith
            _duration = client_runtime + trans_latency + server_runtime + self.server_lib_runtime + self.client_lib_runtime
            durations[i] = _duration

        return durations

    def search_exitpoint_bidirection(self, bandwidth, batch_size=None, min_latency=None):

        durations = np.zeros(shape=(self.n_layers, self.n_layers))

        for i in range(0, self.n_layers):  # outbound
            for j in range(i, self.n_layers):  # in-bound

                if batch_size is None:
                    client_runtime_1 = self.client_profile['runtime'].iloc[0: i].sum()
                    dsize_out = self.dsize['dsize'].iloc[i] / bandwidth
                    server_runtime = self.server_profile['runtime'].iloc[i:j].sum()
                    dsize_in = self.dsize['dsize'].iloc[j] / bandwidth
                    client_runtime_2 = self.client_profile['runtime'].iloc[j:].sum()

                else:
                    client_runtime_1 = self.client_profile['runtime'].iloc[0: i].sum() * batch_size
                    dsize_out = self.dsize['dsize'].iloc[i] * batch_size / bandwidth
                    server_runtime = self.server_profile['runtime'].iloc[i:j].sum() * batch_size
                    dsize_in = self.dsize['dsize'].iloc[j] * batch_size / bandwidth
                    client_runtime_2 = self.client_profile['runtime'].iloc[j:].sum() * batch_size

                _duration = client_runtime_1 + dsize_out + server_runtime + dsize_in + client_runtime_2  \
                            + self.server_lib_runtime + self.client_lib_runtime
                durations[i, j] = _duration

        return durations


def read_txt(file):
    with open(file) as f:
        c = f.readlines()

    return [float(txt[:-2]) for txt in c]