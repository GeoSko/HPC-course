from sklearn.preprocessing import StandardScaler
import h5py
import modules.utils.utils as ut

hf = h5py.File("data.h5", "r")
data = hf["data"]

# fit the StandardScaler from scikit-learn

scaler = StandardScaler()
scaler.fit(data)

# print the results
ut.scaler_resuls(scaler)

hf.close()