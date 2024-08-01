import h5py
import numpy as np
from concurrent import futures
import random
from torch.utils import data
import torch as torch
import time
from train_eval_config.OneConfig import ModelConfig as Config
import os
import glob
import multiprocessing as mp
from multiprocessing import Process, Queue, shared_memory, Event
import queue


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
            print(f"Split data {sidx}:{sidx+max_size} to {fid} part")
            split_fname = hdf5_fname.replace('.hdf5', f'_split{fid}.hdf5')
            if overwrite and os.path.exists(split_fname):
                os.remove(split_fname)
            with h5py.File(split_fname, 'w-') as f_out:
                print(f"save data to {split_fname}")
                for k, v in raw_data.items():
                    f_out.create_dataset(k, data=v[sidx:sidx+max_size], compression="gzip")
            fid += 1




class ParallelLargeDatasets(object):
    
    def __init__(self, replay_dirs, 
                 batch_size, 
                 lstm_steps, device, 
                 train_step_per_buffer, num_workers, max_step, # useless params
                 dataset_name) -> None:
        data_queue = Queue(maxsize=50)
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]
            
        self.data_queue = data_queue
        self.device = device
        self.dataset_process = [mp.Process(target=run_large_datasets, args=(replay_dirs, batch_size, lstm_steps, 
                                            "cpu", train_step_per_buffer, 
                                            num_workers, max_step, dname,
                                            data_queue,
                                            )) for dname in dataset_name]
        [p.start() for p in self.dataset_process]
    
    def __del__(self):
        [p.terminate() for p in self.dataset_process]
        [p.join() for p in self.dataset_process]
    
    def next_batch(self):
        batch = self.data_queue.get()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch

def run_large_datasets(replay_dirs, 
                 batch_size, 
                 lstm_steps, device, 
                 train_step_per_buffer, num_workers, max_step, # useless params
                 dataset_name, data_queue):
    dataset = LargeDatasets(replay_dirs, batch_size, lstm_steps, 
                            device, train_step_per_buffer, 
                            num_workers, max_step, dataset_name,
                            )
    torch.set_num_threads(1)

    # batch = dataset.next_batch()
    while True:
        while not data_queue.full():
            batch = dataset.next_batch()
            data_queue.put(batch)
        time.sleep(0.01)

class LargeDatasets(object):
    """
        Large dastasets, load one data file at a time
    """
    def __init__(self, replay_dirs, 
                 batch_size, 
                 lstm_steps, device, 
                 train_step_per_buffer, num_workers, max_step, # useless params
                 dataset_name) -> None:
        self.replay_dirs = replay_dirs
        self.batch_size = batch_size
        self.lstm_steps = lstm_steps
        self.device = device
        self.train_step = 0
        self.max_step = max_step
        self.change_step = 50000
        self.first_half = True
        self.dataset_name = dataset_name
        
        self.buffered_data_queue = queue.Queue()
        self.buffer_data_num = 2

        # all_keys = [
        #     'observation',
        #     'action',
        #     'reward',
        #     'done',
        #     'legal_action'
        #     'sub_action'
        # ]
        self.data_split = [
            Config.SERI_VEC_SPLIT_SHAPE[0][0],
            len(Config.LABEL_SIZE_LIST),
            1,
            1,
            sum(Config.LEGAL_ACTION_SIZE_LIST),
            len(Config.LABEL_SIZE_LIST),
        ]
        self.done_index = Config.SERI_VEC_SPLIT_SHAPE[0][0] + len(Config.LABEL_SIZE_LIST) + 1 + 1 - 1
        find_path = os.path.join(self.replay_dirs, self.dataset_name, '*.hdf5')
        self.data_files = glob.glob(find_path)
        self.data_files.sort()
        print(f"Founds {len(self.data_files)} data files in {find_path}")
        self.data_id = -1
        self.sample_id = 0
        self.load_dataset()

    def _load_dataset(self):
        """ Load data from files in order """
        self.data_id = (self.data_id + 1) % len(self.data_files)
        print(f"Load data file {self.data_id}, {self.data_files[self.data_id]}")
        data_path = self.data_files[self.data_id]
        with h5py.File(data_path, 'r') as f:
            if "datas" in f.keys():
                num_data = f['datas'].shape[0]
                if num_data < self.batch_size:
                    return None, 0, []
                obs, act, reward, done, legal_act, sub_act = torch.split(torch.tensor(f['datas'][()], dtype=torch.float32), self.data_split, dim=1)
                all_data = {
                    'observation': obs,
                    'action': act,
                    'reward': reward,
                    'done': done,
                    'legal_action': legal_act,
                    'sub_action': sub_act
                }
            else:
                num_data = f['observation'].shape[0]
                if num_data < self.batch_size:
                    return None, 0, []
                all_data = {k: torch.tensor(v[()], dtype=torch.float32) for k, v in f.items()}

            # del raw_data
        print(f"Load data from {data_path}, with number {num_data}.")

        if 'sub_task' in self.replay_dirs:
            # load_index = list(range(total_shape))
            assert False, "Not implemented for sub_task"

        # self.data_length = num_data

        # create random indices
        data_index = list(range(num_data - self.lstm_steps))
        random.shuffle(data_index)
        data_index = torch.tensor(data_index, dtype=torch.long)
        self.sample_id = 0
        return all_data, num_data, data_index
    
    def load_dataset(self):
        # load enough dataset into data_queue
        while self.buffered_data_queue.qsize() < self.buffer_data_num:
            data, num_data, data_index = self._load_dataset()
            if data is not None:
                self.buffered_data_queue.put((data, num_data, data_index))
            
        # get the first dataset from queue
        self.all_data, self.data_length, self.data_index = self.buffered_data_queue.get()

    def next_batch(self):
        # index = random.sample(self.data_index, self.batch_size)  ### 128 ###
        # print("get next batch", self.sample_id, len(self.data_index)) 
        if self.sample_id + self.batch_size >= len(self.data_index):
            self.load_dataset()

        index = self.data_index[self.sample_id : self.sample_id + self.batch_size]
        final_index = []
        lstm_shift = torch.arange(self.lstm_steps)
        for ii in index:  ### revise the chosen index ###
            if torch.sum(self.all_data["done"][ii : ii + self.lstm_steps]) > 0:
                while self.all_data["done"][ii + self.lstm_steps - 1] == 0:
                    ii -= 1

            final_index.extend((lstm_shift + ii).tolist())
        
        final_next_index = (1 + np.array(final_index)).clip(max=self.data_length - 1)

        self.train_step += 1

        batch_data = {}
        for k, v in self.all_data.items():
            batch_data[k] = v[final_index].to(self.device)
        batch_data["next_observation"] = self.all_data["observation"][final_next_index].to(self.device)
        batch_data["next_legal_action"] = self.all_data["legal_action"][final_next_index].to(self.device)

        self.sample_id = self.sample_id + self.batch_size

        return batch_data


if __name__ == "__main__":
    # fpath = "hok1v1/datasets/hard_medium"
    fpath = "hok1v1/datasets/1v1version1/split/"
    fpath = "hok1v1/datasets/1v1version1/"
    dataset = ParallelLargeDatasets(fpath, batch_size=1000, lstm_steps=4, 
                            device="cpu", train_step_per_buffer=1000, num_workers=1, 
                            max_step=500000, dataset_name="test")

    for i in range(10):
        b = dataset.next_batch()
        print(f"get {i} batch, shape", b["done"].shape)

    dataset= None