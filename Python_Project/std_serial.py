from sklearn.preprocessing import StandardScaler
import h5py
from modules.utils.utils import scaler_resuls

hf = h5py.File("data.h5", "r")
data = hf["data"]

# fit the StandardScaler from scikit-learn

scaler = StandardScaler()
scaler.fit(data)

# print the results
scaler_resuls(scaler,"std")

hf.close()