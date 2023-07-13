import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py
import modules.utils.utils as ut

np.random.seed(0)
data = np.random.rand(10000000, 35)

hf = h5py.File("data.h5", "w")
hf.create_dataset('data', data=data)
hf.close()

# fit the StandardScaler from scikit-learn

scaler = StandardScaler()
scaler.fit(data)

# print the results
ut.scaler_resuls(scaler)


# partially fir the StandardScaler

scaler = StandardScaler()
n = data.shape[0] # number of rows
batch_size = 100  # number of rows in each call to partial_fit
index = 0         # helper-var

while index < n:
    partial_size = min(batch_size, n - index)  # needed because last loop is possibly incomplete
    partial_data = data[index:index+partial_size]
    scaler.partial_fit(partial_data)
    index += partial_size

# print the results
ut.scaler_resuls(scaler)

