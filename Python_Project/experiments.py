from modules.parallel_scalers.parallelScalers import ParMinMaxScaler
from modules.parallel_scalers.parallelScalers import ParStandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import h5py
import time

min_max_parallel_times = []
std_parallel_times = []
num_workers = [1, 2, 4, 8, 16]
test_times = 10


# my_times = [20, 12, 7, 4, 3]
# my_serial = 19


# plt.plot(num_workers, my_times, "-o", label="Parallel fitting")
# plt.xticks(num_workers)
# x0 = 1
# y0 = my_serial
# plt.plot(x0, y0, "s", label="Serial fitting")
# plt.xlabel('#workers')
# plt.ylabel('time(s)')
# plt.suptitle('MinMax Parallel Scaler Fitting time')
# plt.legend()
# plt.show()
# plt.savefig('foo.png')
# plt.clf()

# Test minmax serial fit
hf = h5py.File("data.h5", "r")
data = hf["data"]
print("Fitting serial MinMax Scaler")
start = time.time()
for iter in range(test_times):
    scaler = MinMaxScaler()
    scaler.fit(data)
end = time.time()
min_max_serial_time = (end - start)/test_times
print(f"Serial MinMax Scaler average time: {min_max_serial_time} seconds"  )


# Test minmax parallel fit
for workers in num_workers:
    print(f"Fitting parallel MinMax Scaler with {workers} workers")
    start = time.time()
    for iter in range(test_times): 
        scaler = ParMinMaxScaler()
        scaler.parallel_fit(data_file="data.h5", num_workers = workers)
    end = time.time()
    avg_time = (end - start)/test_times
    print(f"Parallel MinMax Scaler({workers} workers) average time: {avg_time} seconds"  )

    min_max_parallel_times.append(avg_time)

print(min_max_serial_time)
print(min_max_parallel_times)


plt.plot(num_workers, min_max_parallel_times, "-o", label="Parallel fitting")
plt.xticks(num_workers)
x0 = 1
y0 = min_max_serial_time
plt.plot(x0, y0, "s", label="Serial fitting")
plt.xlabel('#workers')
plt.ylabel('time(s)')
plt.suptitle('Parallel MinMax Scaler Fitting time')
plt.legend()
plt.show()
plt.savefig('min_max_time.png')
plt.clf()


first_elemet = min_max_parallel_times[0]

min_max_speedup = [first_elemet/element for element in min_max_parallel_times]

# plot speedup

plt.plot(num_workers, min_max_speedup, "-o", label="Parallel speedup")
plt.plot(num_workers, num_workers, "-o", label="Ideal speedup")
plt.xticks(num_workers)
plt.xlabel('#workers')
plt.ylabel('time(s)')
plt.suptitle('Parallel MinMax Scaler Fitting speedup')
plt.legend()
plt.show()
plt.savefig('min_max_speedup.png')
plt.clf()


#########################################################################################################3

# Test std serial fit
hf = h5py.File("data.h5", "r")
data = hf["data"]
print("Fitting serial Standard Scaler")
start = time.time()
for iter in range(test_times):
    scaler = StandardScaler()
    scaler.fit(data)
end = time.time()
std_serial_time = (end - start)/test_times
print(f"Serial Standard Scaler average time: {std_serial_time} seconds"  )

# Test std parallel fit
for workers in num_workers:
    print(f"Fitting parallel Standard Scaler with {workers} workers")
    start = time.time()
    for iter in range(test_times): 
        scaler = ParStandardScaler()
        scaler.parallel_fit(data_file="data.h5", num_workers = workers)
    end = time.time()
    avg_time = (end - start)/test_times
    print(f"Parallel Standard Scaler({workers} workers) average time: {avg_time} seconds"  )
    std_parallel_times.append(avg_time)

print(std_serial_time)
print(std_parallel_times)

plt.plot(num_workers, std_parallel_times, "-o", label="Parallel fitting")
plt.xticks(num_workers)
x0 = 1
y0 = std_serial_time
plt.plot(x0, y0, "s", label="Serial fitting")
plt.xlabel('#workers')
plt.ylabel('time(s)')
plt.suptitle('Parallel Standard Scaler Fitting time')
plt.legend()
plt.show()
plt.savefig('std_time.png')
plt.clf()


first_elemet = std_parallel_times[0]

std_speedup = [first_elemet/element for element in std_parallel_times]

# plot speedup

plt.plot(num_workers, std_speedup, "-o", label="Parallel speedup")
plt.plot(num_workers, num_workers, "-o", label="Ideal speedup")
plt.xticks(num_workers)
plt.xlabel('#workers')
plt.ylabel('time(s)')
plt.suptitle('Parallel MinMax Scaler Fitting speedup')
plt.legend()
plt.show()
plt.savefig('std_speedup.png')
plt.clf()
