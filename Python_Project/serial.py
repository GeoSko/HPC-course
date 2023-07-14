from sklearn.preprocessing import StandardScaler
import h5py

hf = h5py.File("data.h5", "r")
data = hf["data"]

# fit the StandardScaler from scikit-learn

scaler = StandardScaler()
scaler.fit(data)

# print the results

hf.close()