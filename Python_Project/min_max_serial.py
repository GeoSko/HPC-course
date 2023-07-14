from sklearn.preprocessing import MinMaxScaler
import h5py
from modules.utils.utils import scaler_resuls

hf = h5py.File("data.h5", "r")
data = hf["data"]

# fit the MinMaxScaler from scikit-learn

scaler = MinMaxScaler()
scaler.fit(data)

# print the results
scaler_resuls(scaler,"min_max")

hf.close()