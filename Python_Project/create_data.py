import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py
import modules.utils.utils as ut

np.random.seed(0)
data = np.random.rand(10000000, 30)

hf = h5py.File("data.h5", "w")
hf.create_dataset('data', data=data)
hf.close()