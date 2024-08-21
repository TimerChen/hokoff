import h5py
import time
import random
import os
# from .large_datasets import split_data

def split_data(hdf5_fname, max_size=500000, overwrite=False):
    """
    Split data into two parts
    """
    
    with h5py.File(hdf5_fname, 'r') as f:
        keys = list(f.keys())
        num_data = f[keys[0]].shape[0]
        raw_data = {k: f[k][()] for k in keys}
        fid = 0
        
        for sidx in range(0, num_data, max_size):
            # Generate the split file name
            split_fname = hdf5_fname.replace('.hdf5', f'_split{fid}.hdf5')
            
            # Check if the file already exists and increment fid if necessary
            while os.path.exists(split_fname):
                fid += 1
                split_fname = hdf5_fname.replace('.hdf5', f'_split{fid}.hdf5')
                
            print(f"Split data {sidx}:{sidx+max_size} to {fid} part")
            if overwrite and os.path.exists(split_fname):
                os.remove(split_fname)  # This line is now less relevant due to the check
            
            with h5py.File(split_fname, 'w-') as f_out:
                print(f"Save data to {split_fname}")
                for k, v in raw_data.items():
                    f_out.create_dataset(k, data=v[sidx:sidx+max_size], compression="gzip")
            fid += 1
        

# if __name__ == "__main__":
#     data_path = "/home/jbhu/nas/code/sample/hokoff/hok1v1/datasets/collection_1vs1/1vs1_0.hdf5"
#     # data_path = "hok1v1/datasets/1v1version1/tensorflow/0_0_split1.hdf5"
#     # data_path = "/NAS2020/Share/jxchen/hok1_datasets/1vs1_0.hdf5"
#     f = h5py.File(data_path, 'r')
#     ks = list(f.keys())
#     print("f", f.keys(), f[ks[0]].shape)
#     split_data(data_path, max_size=500000, overwrite=True)
    
if __name__ == "__main__":
    # Loop through the range of files from 1vs1_0.hdf5 to 1vs1_9.hdf5
    data_path = "/home/jbhu/nas/code/sample/hokoff/hok1v1/datasets/collection_1vs1/1vs1_9.hdf5"
    print(f"Processing {data_path}...")
    split_data(data_path, max_size=500000)