import time
import pickle
import numpy as np
import os
from paths import *
from misc_tools import PathUtils as PU



if __name__ == '__main__': 
    conduct_path = src_ + "debug/"
    many_path = conduct_path + "many/"
    PU.mk(conduct_path)
    PU.mk(many_path)
    big_name = os.path.join(conduct_path, "bigdata.feat")

    # Generate some example data (list of NumPy arrays)
    data_list = [np.random.rand(15, 39) for _ in range(1000000)]

    # Save the data using pickle
    with open(big_name, 'wb') as file:
        pickle.dump(data_list, file)

    # Save each NumPy array as a separate file
    for i, arr in enumerate(data_list):
        np.save(many_path + f'/array_{i}.npy', arr)

    # Load data using pickle and measure time
    start_time = time.time()
    with open(big_name, 'rb') as file:
        loaded_data_pkl = pickle.load(file)
    loading_time_pkl = time.time() - start_time

    # Load data from individual NumPy array files and measure time
    loaded_data_files = []
    start_time = time.time()
    for i in range(len(data_list)):
        loaded_arr = np.load(many_path + f'/array_{i}.npy')
        loaded_data_files.append(loaded_arr)
    loading_time_files = time.time() - start_time

    # Compare loading times
    print(f"Loading time using pickle: {loading_time_pkl:.4f} seconds")
    print(f"Loading time from individual files: {loading_time_files:.4f} seconds")

    # Check if the loaded data is the same
    data_equal = all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(loaded_data_pkl, loaded_data_files))
    if data_equal:
        print("Data loaded using both methods is identical.")
    else:
        print("Data loaded using both methods is not identical.")
