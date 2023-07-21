from modules.parallel_scalers.parallelScalers import ParStandardScaler



# hf = h5py.File("data.h5", "r")
# data = hf["data"]
scaler = ParStandardScaler()
scaler.parallel_fit(data_file="data.h5", num_workers = 8)

print(scaler)
