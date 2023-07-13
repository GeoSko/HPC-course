import pandas as pd
from sklearn import preprocessing
from modules.utils.utils_standardscaler import reduce_scalers
import h5py
from mpi4py.futures import MPIPoolExecutor
from concurrent.futures import ProcessPoolExecutor



def work(data):
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    return scaler

__all__ = [
    'ParStandardScaler'
]



def _copy_attr(target_obj, source_obj):
    for attr in vars(source_obj):
        setattr(target_obj, attr, getattr(source_obj, attr))


class ParStandardScaler(preprocessing.StandardScaler):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(ParStandardScaler, self).__init__(
            copy=copy, with_mean=with_mean, with_std=True)

    def parallel_fit(self, data_file, num_idxs, sample_weight=None, chunksize=100):
        # data = pd.read_csv(data_file, chunksize=chunksize)
        # scalers = []
        # chunk-based approach that can be parallelized
        # for chunk in data:
            
        #     scaler = preprocessing.StandardScaler()
        #     scaler.fit(chunk[num_idxs].values)
        #     scalers.append(scaler)

        ###########
        
        hf = h5py.File(data_file, "r")
        data = hf["data"]
        scalers = []
        n = data.shape[0] # number of rows
        batch_size = 5000000  # number of rows in each call to partial_fit
        index = 0         # helper-var
        iterations = int(n / batch_size)

        print(iterations)
        # for i in range(0, iterations):
        #     scaler = preprocessing.StandardScaler()
        #     partial_data = data[index:index+batch_size]
        #     scaler.fit(partial_data)
        #     index += batch_size
        #     scalers.append(scaler)

        all_data = [data[start*batch_size:(start*batch_size)+batch_size] for start in range(iterations)]

        # with ThreadPoolExecutor(max_workers=4) as executor:
        with ProcessPoolExecutor(max_workers=4) as executor:
            scalers = executor.map(work, all_data)
            # scalers.append(scaler)

        # executor = ProcessPoolExecutor(max_workers=4)
        # for iter in range(iterations):
        #     scaler = executor.submit(work, data[iter*batch_size:(iter*batch_size)+batch_size])
        #     scalers.append(scaler)

                

        # print("HELLO",all_data[0].shape)
        # with MPIPoolExecutor() as executor:
        #     scaler = executor.map(work, all_data)
        #     scalers.append(scaler)

        # while index < n:
        #     scaler = preprocessing.StandardScaler()
        #     partial_size = min(batch_size, n - index)  # needed because last loop is possibly incomplete
        #     partial_data = data[index:index+partial_size]
        #     # scaler.partial_fit(partial_data)
        #     scaler.fit(partial_data)
        #     index += partial_size
        #     scalers.append(scaler)
        ###########
        scalers = list(scalers)

        remaining = n%batch_size
        if( remaining != 0):
            print(remaining)
            scaler = preprocessing.StandardScaler()
            start = iterations*batch_size
            partial_data = data[start:start+remaining]
            scaler.fit(partial_data)
            scalers.append(scaler)


        # print("INFO: number of chunks/scalers=", len(scalers))
        final_scaler = reduce_scalers(scalers)

        _copy_attr(self, final_scaler)

        del(final_scaler)

