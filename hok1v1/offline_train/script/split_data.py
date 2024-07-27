import h5py
import time
import random
import os
from large_datasets import split_data
        

if __name__ == "__main__":
    data_path = "hok1v1/datasets/hard_medium/all_data/all_data.hdf5"
    # data_path = "hok1v1/datasets/1v1version1/tensorflow/0_0_split1.hdf5"
    # data_path = "/NAS2020/Share/jxchen/hok1_datasets/1vs1_0.hdf5"
    f = h5py.File(data_path, 'r')
    ks = list(f.keys())
    print("f", f.keys(), f[ks[0]].shape)
    split_data(data_path, max_size=500000, overwrite=True)