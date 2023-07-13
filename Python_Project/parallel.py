from modules.utils.utils import scaler_resuls
from modules.parallel_scalers.parallelstandardscaler import ParStandardScaler
import h5py
import modules.utils.utils as ut



# hf = h5py.File("data.h5", "r")
# data = hf["data"]
scaler = ParStandardScaler()
scaler.parallel_fit(data_file="data.h5", num_idxs=10)
ut.scaler_resuls(scaler)
