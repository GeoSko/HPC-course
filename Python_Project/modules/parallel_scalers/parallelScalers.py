import pandas as pd
from sklearn import preprocessing
from modules.utils.utils_StandardScaler import reduce_scalers as reduce_std
from modules.utils.utils_MinMaxScaler import reduce_scalers as reduce_min_max

import h5py
from mpi4py.futures import MPIPoolExecutor
from concurrent.futures import ProcessPoolExecutor



def work_std(attributs):
    start = attributs[0]
    end = attributs[1]
    data_file = attributs[2]
    data_file = "./" + data_file
    hf = h5py.File(data_file, "r")
    data = hf["data"]
    partial_data = data[start:end]
    scaler = preprocessing.StandardScaler()
    scaler.partial_fit(partial_data)

    return scaler

def work_min_max(attriburs):
    start = attriburs[0]
    end = attriburs[1]
    data_file = attriburs[2]
    data_file = "./" + data_file
    hf = h5py.File(data_file, "r")
    data = hf["data"]
    partial_data = data[start:end]
    scaler = preprocessing.MinMaxScaler()
    scaler.partial_fit(partial_data)

    return scaler

__all__ = [
    'ParStandardScaler','ParMinMaxScaler'
]



def _copy_attr(target_obj, source_obj):
    for attr in vars(source_obj):
        setattr(target_obj, attr, getattr(source_obj, attr))


class ParStandardScaler(preprocessing.StandardScaler):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(ParStandardScaler, self).__init__(
            copy=copy, with_mean=with_mean, with_std=True)
        
    def __str__(self) -> str:
        attributes = '==> StandardScaler <==\n'
        for attr in vars(self):
            attributes += "scaler.{} = {}\n".format(attr, getattr(self, attr))
        return str(attributes)

    def parallel_fit(self, data_file, num_workers):        
        hf = h5py.File(data_file, "r")
        data = hf["data"]
        scalers = []
        n = data.shape[0] # number of rows
        batch_size = int(n/num_workers)
        iterations = int(n / batch_size)

        print(f"Parallel fitting Standard Scaler using\nWorkers:{num_workers}\nbatch_size:{batch_size}\n")

        attributs = [(start*batch_size,(start*batch_size)+batch_size, data_file) for start in range(iterations)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            scalers = executor.map(work_std, attributs)

        
        scalers = list(scalers)

        # print("INFO: number of chunks/scalers=", len(scalers))
        final_scaler = reduce_std(scalers)

        #Fit remaining datapoints that did not divide with batch_size
        remaining = n%batch_size
        if( remaining != 0):
            print(f"Remaining {remaining} datapoints\n")
            start = iterations*batch_size
            partial_data = data[start:start+remaining]
            final_scaler.partial_fit(partial_data)
        hf.close()

        _copy_attr(self, final_scaler)

        del(final_scaler)

class ParMinMaxScaler(preprocessing.MinMaxScaler):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(ParMinMaxScaler, self).__init__(
            copy=copy)
        
    def __str__(self) -> str:
        attributes = '==> MinMaxScaler <==\n'
        for attr in vars(self):
            attributes += "scaler.{} = {}\n".format(attr, getattr(self, attr))
        return str(attributes)

    def parallel_fit(self, data_file, num_workers):        
        hf = h5py.File(data_file, "r")
        data = hf["data"]
        scalers = []
        n = data.shape[0] # number of rows
        batch_size = int(n/num_workers)
        iterations = int(n / batch_size)

        print(f"Parallel fitting MinMax Scaler using\nWorkers:{num_workers}\nbatch_size:{batch_size}\n")

        attriburs = [(start*batch_size,(start*batch_size)+batch_size, data_file) for start in range(iterations)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            scalers = executor.map(work_min_max, attriburs)

        
        scalers = list(scalers)

        # print("INFO: number of chunks/scalers=", len(scalers))
        final_scaler = reduce_min_max(scalers)

        #Fit remaining datapoints that did not divide with batch_size
        remaining = n%batch_size
        if( remaining != 0):
            print(f"Remaining {remaining} datapoints\n")
            start = iterations*batch_size
            partial_data = data[start:start+remaining]
            final_scaler.partial_fit(partial_data)
        hf.close()

        _copy_attr(self, final_scaler)

        del(final_scaler)

