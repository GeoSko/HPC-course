import numpy as np
import h5py

np.random.seed(0)
data = np.random.rand(30000000, 30)

# data = np.random.rand(2, 2)

hf = h5py.File("data.h5", "w")
hf.create_dataset('data', data=data)
hf.close()