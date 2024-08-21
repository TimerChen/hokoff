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
import multiprocessing
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
    def __init__(self, replay_dirs, batch_size, lstm_steps, device, train_step_per_buffer, num_workers, max_step, dataset_name):
        print((replay_dirs, batch_size, lstm_steps, device, train_step_per_buffer, num_workers, max_step, dataset_name))
        print("pppppp")
        data_queue = multiprocessing.Queue(maxsize=100000000)
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]
        
        self.data_queue = data_queue
        self.device = device
        
        self.dataset_process = []
        self.lock = multiprocessing.Lock()  # 创建锁
        
        data_path = "/NAS2020/Workspaces/DRLGroup/jbhu/code/sample/hokoff/hok1v1/datasets/collection_1vs1/1vs1_0_split0.hdf5"
        
        with h5py.File(data_path, 'r') as f:
            print("bug1")
            print(f.keys())
            if "datas" in f.keys():
                print("bug2")
                num_data = f['datas'].shape[0]
                # if num_data < self.batch_size:
                #     return None, 0, []
                obs, act, reward, done, legal_act, sub_act = torch.split(torch.tensor(f['datas'][()], dtype=torch.float32), self.data_split, dim=1)
                all_data = {
                    'observation': obs,
                    'action': act,
                    'reward': reward,
                    'done': done,
                    'legal_action': legal_act,
                    'sub_action': sub_act
                }
                print("bug3")
            else:
                print("bug4")
                num_data = f['observation'].shape[0]
                print("b")
                print(num_data)
                # if num_data < self.batch_size:
                #     return None, 0, []
                print("bug5")
                tensor = torch.tensor([1, 2, 3])
                # self.lock.acquire()
                print(tensor)
                # try:
                #     tensor1 = torch.tensor(f['observation'][:], dtype=torch.float32)
                # except:
                #     print("error")
                # tensor2 = torch.tensor(f['action'][:], dtype=torch.float32)
                # tensor3 = torch.tensor(f['reward'][:], dtype=torch.float32)
                # tensor4 = torch.tensor(f['done'][:], dtype=torch.float32)
                # tensor5 = torch.tensor(f['legal_action'][:], dtype=torch.float32)
                # tensor6 = torch.tensor(f['sub_action'][:], dtype=torch.float32)
                    
                self.all_data_f = h5py.File(data_path, 'r')
                # print(f"Opened HDF5 file: {filename} successfully")
                    
                    # 读取数据并转换为张量
                tensor1 = torch.tensor(self.all_data_f['observation'][:], dtype=torch.float32)
                tensor2 = torch.tensor(self.all_data_f['action'][:], dtype=torch.float32)
                tensor3 = torch.tensor(self.all_data_f['reward'][:], dtype=torch.float32)
                tensor4 = torch.tensor(self.all_data_f['done'][:], dtype=torch.float32)
                tensor5 = torch.tensor(self.all_data_f['legal_action'][:], dtype=torch.float32)
                tensor6 = torch.tensor(self.all_data_f['sub_action'][:], dtype=torch.float32)
                    
                print(tensor1.shape)
                print(tensor2.shape)
                print(tensor3.shape)
                print(tensor4.shape)
                print(tensor5.shape)
                print(tensor6.shape)
                print(tensor1.shape)
                    # print(tensor2.shape)
                    # print(tensor3.shape)
                    # print(tensor4.shape)
                    # print(tensor5.shape)
                    # print(tensor6.shape)

                    # 这里可以将读取的数据合并到 self.all_data 中
                    # 例如：
                all_data = torch.cat((self.all_data, tensor1, tensor2, tensor3, tensor4, tensor5, tensor6), dim=0)
                # self.lock.release()
                # all_data = {k: torch.tensor(v[()], dtype=torch.float32) for k, v in f.items()}

            # del raw_data
        print(f"Load data from {data_path}, with number {num_data}.")

        if 'sub_task' in self.replay_dirs:
            # load_index = list(range(total_shape))
            assert False, "Not implemented for sub_task"

        # self.data_length = num_data

        # create random indices
        print("out")
        data_index = list(range(num_data - self.lstm_steps))
        random.shuffle(data_index)
        data_index = torch.tensor(data_index, dtype=torch.long)
        self.sample_id = 0
        print("all")
        print(all_data)
        return all_data, num_data, data_index
        
        for dname in dataset_name:
            process = multiprocessing.Process(target=self.run_large_datasets, args=(replay_dirs, batch_size, lstm_steps, "cpu", train_step_per_buffer, num_workers, max_step, dname, data_queue))
            self.dataset_process.append(process)
            process.start()
            process.join()
            
    
    def __del__(self):
        for process in self.dataset_process:
            process.terminate()
            process.join()
    
    def next_batch(self):
        with self.lock:
            print("size")
            print(self.data_queue.qsize())
            batch = self.data_queue.get()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            return batch
    
    def run_large_datasets(self, replay_dirs, batch_size, lstm_steps, device, train_step_per_buffer, num_workers, max_step, dataset_name, data_queue):
        dataset = LargeDatasets(replay_dirs, batch_size, lstm_steps, device, train_step_per_buffer, num_workers, max_step, dataset_name)
        print("DATASETS")
        torch.set_num_threads(1)

        while True:
            while not data_queue.full():
                print("popop")
                batch = dataset.next_batch()
                data_queue.put(batch)
            time.sleep(0.01)


# class ParallelLargeDatasets(object):
    
#     def __init__(self, replay_dirs, 
#                  batch_size, 
#                  lstm_steps, device, 
#                  train_step_per_buffer, num_workers, max_step, # useless params
#                  dataset_name) -> None:
#         data_queue = Queue(maxsize=100000)
#         if isinstance(dataset_name, str):
#             dataset_name = [dataset_name]
            
#         self.data_queue = data_queue
#         self.device = device
        
#         self.dataset_process = [mp.Process(target=run_large_datasets, args=(replay_dirs, batch_size, lstm_steps, 
#                                             "cpu", train_step_per_buffer, 
#                                             num_workers, max_step, dname,
#                                             data_queue,
#                                             )) for dname in dataset_name]
#         [p.start() for p in self.dataset_process]
    
#     def __del__(self):
#         [p.terminate() for p in self.dataset_process]
#         [p.join() for p in self.dataset_process]
    
#     def next_batch(self):
#         print("size")
#         print(self.data_queue.qsize())
#         batch = self.data_queue.get()
#         batch = {k: v.to(self.device) for k, v in batch.items()}
#         return batch

# def run_large_datasets(replay_dirs, 
#                  batch_size, 
#                  lstm_steps, device, 
#                  train_step_per_buffer, num_workers, max_step, # useless params
#                  dataset_name, data_queue):

#     dataset = LargeDatasets(replay_dirs, batch_size, lstm_steps, 
#                             device, train_step_per_buffer, 
#                             num_workers, max_step, dataset_name,
#                             )
#     print("DATASETS")
#     torch.set_num_threads(1)

#     # batch = dataset.next_batch()
#     while True:
#         while not data_queue.full():
#             print("popop")
#             batch = dataset.next_batch()
#             data_queue.put(batch)
#         time.sleep(0.01)

class LargeDatasets(object):
    """
        Large dastasets, load one data file at a time
    """
    def __init__(self, replay_dirs, 
                 batch_size, 
                 lstm_steps, device, 
                 train_step_per_buffer, num_workers, max_step, # useless params
                 dataset_name) -> None:
        print("LD")
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
        self.lock = multiprocessing.Lock()

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
        # self._load_dataset()
        self.load_dataset()

    def _load_dataset(self):
        print("LOAD")
        """ Load data from files in order """
        self.data_id = (self.data_id + 1) % len(self.data_files)
        print(f"Load data file {self.data_id}, {self.data_files[self.data_id]}")
        data_path = self.data_files[self.data_id]
        with h5py.File(data_path, 'r') as f:
            print("bug1")
            print(f.keys())
            if "datas" in f.keys():
                print("bug2")
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
                print("bug3")
            else:
                print("bug4")
                num_data = f['observation'].shape[0]
                print("b")
                print(num_data)
                if num_data < self.batch_size:
                    return None, 0, []
                print("bug5")
                tensor = torch.tensor([1, 2, 3])
                # self.lock.acquire()
                print(tensor)
                try:
                    tensor1 = torch.tensor(f['observation'][:], dtype=torch.float32)
                except:
                    print("error")
                tensor2 = torch.tensor(f['action'][:], dtype=torch.float32)
                tensor3 = torch.tensor(f['reward'][:], dtype=torch.float32)
                tensor4 = torch.tensor(f['done'][:], dtype=torch.float32)
                tensor5 = torch.tensor(f['legal_action'][:], dtype=torch.float32)
                tensor6 = torch.tensor(f['sub_action'][:], dtype=torch.float32)
                    
                print(tensor1.shape)
                    # print(tensor2.shape)
                    # print(tensor3.shape)
                    # print(tensor4.shape)
                    # print(tensor5.shape)
                    # print(tensor6.shape)

                    # 这里可以将读取的数据合并到 self.all_data 中
                    # 例如：
                all_data = torch.cat((self.all_data, tensor1, tensor2, tensor3, tensor4, tensor5, tensor6), dim=0)
                # self.lock.release()
                # all_data = {k: torch.tensor(v[()], dtype=torch.float32) for k, v in f.items()}

            # del raw_data
        print(f"Load data from {data_path}, with number {num_data}.")

        if 'sub_task' in self.replay_dirs:
            # load_index = list(range(total_shape))
            assert False, "Not implemented for sub_task"

        # self.data_length = num_data

        # create random indices
        print("out")
        data_index = list(range(num_data - self.lstm_steps))
        random.shuffle(data_index)
        data_index = torch.tensor(data_index, dtype=torch.long)
        self.sample_id = 0
        print("all")
        print(all_data)
        return all_data, num_data, data_index
    
    def load_dataset(self):
        # load enough dataset into data_queue
        while self.buffered_data_queue.qsize() < self.buffer_data_num:
            print("num")
            print(self.buffer_data_num)
            data, num_data, data_index = self._load_dataset()
            if data is not None:
                self.buffered_data_queue.put((data, num_data, data_index))
                print("123")
                print(self.buffered_data_queue.qsize())
                print("321")
            
        # get the first dataset from queue
        print("ENDLOAD")
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

# import h5py
# import numpy as np
# from concurrent import futures
# import random
# from torch.utils import data
# import torch as torch
# import time
# from train_eval_config.OneConfig import ModelConfig as Config
# import os
# import glob
# import multiprocessing as mp
# from multiprocessing import Process, Queue, Event
# # from shared_numpy import shared_memory
# import queue


# def split_data(hdf5_fname, max_size=500000, overwrite=False):
#     """
#     Split data into two parts
#     """
    
#     with h5py.File(hdf5_fname, 'r') as f:
#         keys = list(f.keys())
#         num_data = f[keys[0]].shape[0]
#         raw_data = {k: f[k][()] for k in keys}
#         fid = 0
#         for sidx in range(0, num_data, max_size):
#             print(f"Split data {sidx}:{sidx+max_size} to {fid} part")
#             split_fname = hdf5_fname.replace('.hdf5', f'_split{fid}.hdf5')
#             if overwrite and os.path.exists(split_fname):
#                 os.remove(split_fname)
#             with h5py.File(split_fname, 'w-') as f_out:
#                 print(f"save data to {split_fname}")
#                 for k, v in raw_data.items():
#                     f_out.create_dataset(k, data=v[sidx:sidx+max_size], compression="gzip")
#             fid += 1




# # class ParallelLargeDatasets(object):
    
# #     def __init__(self, replay_dirs, 
# #                  batch_size, 
# #                  lstm_steps, device, 
# #                  train_step_per_buffer, num_workers, max_step, # useless params
# #                  dataset_name) -> None:
# #         data_queue = Queue(maxsize=50)
# #         if isinstance(dataset_name, str):
# #             dataset_name = [dataset_name]
            
# #         self.data_queue = data_queue
# #         self.device = device
# #         self.dataset_process = [mp.Process(target=run_large_datasets, args=(replay_dirs, batch_size, lstm_steps, 
# #                                             "cpu", train_step_per_buffer, 
# #                                             num_workers, max_step, dname,
# #                                             data_queue,
# #                                             )) for dname in dataset_name]
# #         [p.start() for p in self.dataset_process]
    
# #     def __del__(self):
# #         [p.terminate() for p in self.dataset_process]
# #         [p.join() for p in self.dataset_process]
    
# #     def next_batch(self):
# #         print("111")
# #         batch = self.data_queue.get() 
# #         print("1111")
# #         batch = {k: v.to(self.device) for k, v in batch.items()}
# #         return batch

# # def run_large_datasets(replay_dirs, 
# #                  batch_size, 
# #                  lstm_steps, device, 
# #                  train_step_per_buffer, num_workers, max_step, # useless params
# #                  dataset_name, data_queue):
# #     dataset = LargeDatasets(replay_dirs, batch_size, lstm_steps, 
# #                             device, train_step_per_buffer, 
# #                             num_workers, max_step, dataset_name,
# #                             )
# #     torch.set_num_threads(1)

# #     # batch = dataset.next_batch()
# #     while True:
# #         while not data_queue.full():
            
# #             batch = dataset.next_batch()
# #             data_queue.put(batch)
# #         time.sleep(0.01)

# class ParallelLargeDatasets(object):
    
#     def __init__(self, replay_dirs, 
#                  batch_size, 
#                  lstm_steps, device, 
#                  train_step_per_buffer, num_workers, max_step, # useless params
#                  dataset_name) -> None:
#         data_queue = Queue(maxsize=50)
#         if isinstance(dataset_name, str):
#             dataset_name = [dataset_name]
            
#         self.data_queue = data_queue
#         self.device = device
#         self.dataset_process = [mp.Process(target=run_large_datasets, args=(replay_dirs, batch_size, lstm_steps, 
#                                             "cpu", train_step_per_buffer, 
#                                             num_workers, max_step, dname,
#                                             data_queue,
#                                             )) for dname in dataset_name]
#         [p.start() for p in self.dataset_process]
    
#     def __del__(self):
#         [p.terminate() for p in self.dataset_process]
#         [p.join() for p in self.dataset_process]
    
#     def next_batch(self):
#         batch = self.data_queue.get()
#         batch = {k: v.to(self.device) for k, v in batch.items()}
#         return batch

# def run_large_datasets(replay_dirs, 
#                  batch_size, 
#                  lstm_steps, device, 
#                  train_step_per_buffer, num_workers, max_step, # useless params
#                  dataset_name, data_queue):
#     dataset = LargeDatasets(replay_dirs, batch_size, lstm_steps, 
#                             device, train_step_per_buffer, 
#                             num_workers, max_step, dataset_name,
#                             )
#     torch.set_num_threads(1)

#     # batch = dataset.next_batch()
#     while True:
#         while not data_queue.full():
#             batch = dataset.next_batch()
#             data_queue.put(batch)
#         time.sleep(0.01)

# class LargeDatasets(object):
#     """
#         Large dastasets, load one data file at a time
#     """
#     def __init__(self, replay_dirs, 
#                  batch_size, 
#                  lstm_steps, device, 
#                  train_step_per_buffer, num_workers, max_step, # useless params
#                  dataset_name) -> None:
#         self.replay_dirs = replay_dirs
#         self.batch_size = batch_size
#         self.lstm_steps = lstm_steps
#         self.device = device
#         self.train_step = 0
#         self.max_step = max_step
#         self.change_step = 50000
#         self.first_half = True
#         self.dataset_name = dataset_name
        
#         self.buffered_data_queue = queue.Queue()
#         self.buffer_data_num = 10

#         # all_keys = [
#         #     'observation',
#         #     'action',
#         #     'reward',
#         #     'done',
#         #     'legal_action'
#         #     'sub_action'
#         # ]
#         self.data_split = [
#             Config.SERI_VEC_SPLIT_SHAPE[0][0],
#             len(Config.LABEL_SIZE_LIST),
#             1,
#             1,
#             sum(Config.LEGAL_ACTION_SIZE_LIST),
#             len(Config.LABEL_SIZE_LIST),
#         ]
#         self.done_index = Config.SERI_VEC_SPLIT_SHAPE[0][0] + len(Config.LABEL_SIZE_LIST) + 1 + 1 - 1
#         find_path = os.path.join(self.replay_dirs, self.dataset_name, '*.hdf5')
#         self.data_files = glob.glob(find_path)
#         self.data_files.sort()
#         print(f"Founds {len(self.data_files)} data files in {find_path}")
#         self.data_id = -1
#         self.sample_id = 0
        
        
#         ################################## added
#         print(self.dataset_name)
#         current_folder = os.path.dirname("/NAS2020/Workspaces/DRLGroup/jbhu/code/sample/hokoff/hok1v1/datasets/collection_5vs5/")
#         self.file_paths = []
#         for filename in os.listdir(current_folder):
#             file_path = os.path.join(current_folder, filename)
#             if os.path.isfile(file_path):
#                 self.file_paths.append(file_path)
#         self.file_index = 0
#         print("---")
#         print(self.file_paths)
#         # self.load_all()
#         ##################################
        
#         self.load_dataset()
    
#     # def _load_dataset(self):
#     #     """ Load data from files in order """
                
#     #     path = self.file_paths[self.file_index % len(self.file_paths)]
#     #     print(f"Load data file {path}")
#     #     # data_path = self.data_files[self.data_id]
#     #     with h5py.File(path, 'r') as f:
#     #         print("aubgakfb")
#     #         if "datas" in f.keys():
#     #             print("expected")
#     #             num_data = f['datas'].shape[0]
#     #             obs, act, reward, done, legal_act, sub_act = torch.split(torch.tensor(f['datas'][()]), self.data_split, dim=1)
#     #             self.all_data = torch.cat((obs, act, reward, done, legal_act, sub_act), dim=1)
#     #             self.all_data = self.all_data.to(torch.float32)
#     #             # = {
#     #             #     'observation': obs,
#     #             #     'action': act,
#     #             #     'reward': reward,
#     #             #     'done': done,
#     #             #     'legal_action': legal_act,
#     #             #     'sub_action': sub_act
#     #             # }

#     #         else:
#     #             print("1111")
#     #             print(f.keys())
#     #             for key in f.keys():
#     #                 print(f"{key}: {f[key].shape}")
#     #             try:
#     #                 print("oqiwdhqwoi")
#     #                 tensor2 = torch.tensor(f['action'][:], dtype=torch.float32)
#     #             except:
#     #                 print("why")
#     #             print("oqwihdoiqhwd")
#     #             tensor4 = torch.tensor(f['done'][:], dtype=torch.float32)
#     #             tensor5 = torch.tensor(f['legal_action'][:], dtype=torch.float32)
#     #             tensor1 = torch.tensor(f['observation'][:], dtype=torch.float32)
#     #             tensor3 = torch.tensor(f['reward'][:], dtype=torch.float32)
#     #             tensor6 = torch.tensor(f['sub_action'][:], dtype=torch.float32)
#     #             print(tensor1.shape)
#     #             print(tensor2.shape)
#     #             print(tensor3.shape)
#     #             print(tensor4.shape)
#     #             print(tensor5.shape)
#     #             print(tensor6.shape)
#     #             # 拼接数据
#     #             all_data = torch.cat((tensor1, tensor2, tensor3, tensor4, tensor5, tensor6), dim=1)
#     #             all_data.to(torch.float32)
#     #             # 将拼接后的数据添加到列表中
#     #             self.all_data = all_data
#     #             self.all_data = self.all_data.to(torch.float32)
#     #             num_data = f['observation'].shape[0]
#     #             # self.all_data = {k: torch.tensor(v[()]) for k, v in f.items()}
#     #             # self.all_data = self.all_data.to(torch.float32)

#     #         # del raw_data
#     #     # print(f"Load data from {data_path}, with number {num_data}.")

#         # if 'sub_task' in self.replay_dirs:
#         #     # load_index = list(range(total_shape))
#         #     assert False, "Not implemented for sub_task"

#         # self.data_length = len(self.all_data)

#         # # create random indices
#         # data_index = list(range(self.data_length - self.lstm_steps))
#         # random.shuffle(self.data_index)
#         # self.sample_id = 0
#         # print(all_data)
#         # return all_data, num_data, data_index
    

#     def _load_dataset(self):
#         """ Load data from files in order """
#         self.data_id = (self.data_id + 1) % len(self.data_files)
#         print(f"Load data file {self.data_id}, {self.data_files[self.data_id]}")
#         data_path = self.data_files[self.data_id]
#         with h5py.File(data_path, 'r') as f:
#             if "datas" in f.keys():
#                 num_data = f['datas'].shape[0]
#                 if num_data < self.batch_size:
#                     return None, 0, []
#                 obs, act, reward, done, legal_act, sub_act = torch.split(torch.tensor(f['datas'][()], dtype=torch.float32), self.data_split, dim=1)
#                 all_data = {
#                     'observation': obs,
#                     'action': act,
#                     'reward': reward,
#                     'done': done,
#                     'legal_action': legal_act,
#                     'sub_action': sub_act
#                 }
#             else:
#                 num_data = f['observation'].shape[0]
#                 if num_data < self.batch_size:
#                     return None, 0, []
#                 print("before read")
#                 all_data = {k: torch.tensor(v[()], dtype=torch.float32) for k, v in f.items()} 
#                 print("after read")

#             # del raw_data
#         print(f"Load data from {data_path}, with number {num_data}.")

#         if 'sub_task' in self.replay_dirs:
#             # load_index = list(range(total_shape))
#             assert False, "Not implemented for sub_task"

#         # self.data_length = num_data

#         # create random indices
#         data_index = list(range(num_data - self.lstm_steps))
#         random.shuffle(data_index)
#         data_index = torch.tensor(data_index, dtype=torch.long)
#         self.sample_id = 0
#         return all_data, num_data, data_index
    
#     def load_dataset(self):
#         # load enough dataset into data_queue
#         while self.buffered_data_queue.qsize() < self.buffer_data_num:
#             print("queue expanding")
#             data, num_data, data_index = self._load_dataset()
#             if data is not None:
#                 print("put success")
#                 self.buffered_data_queue.put((data, num_data, data_index))
            
#         # get the first dataset from queue
#         self.all_data, self.data_length, self.data_index = self.buffered_data_queue.get()

#     def next_batch(self):
#         # index = random.sample(self.data_index, self.batch_size)  ### 128 ###
#         # print("get next batch", self.sample_id, len(self.data_index)) 
#         if self.sample_id + self.batch_size >= len(self.data_index):
#             self.load_dataset()

#         index = self.data_index[self.sample_id : self.sample_id + self.batch_size]
#         final_index = []
#         lstm_shift = torch.arange(self.lstm_steps)
#         for ii in index:  ### revise the chosen index ###
#             if torch.sum(self.all_data["done"][ii : ii + self.lstm_steps]) > 0:
#                 while self.all_data["done"][ii + self.lstm_steps - 1] == 0:
#                     ii -= 1

#             final_index.extend((lstm_shift + ii).tolist())
        
#         final_next_index = (1 + np.array(final_index)).clip(max=self.data_length - 1)

#         self.train_step += 1

#         batch_data = {}
#         for k, v in self.all_data.items():
#             batch_data[k] = v[final_index].to(self.device)
#         batch_data["next_observation"] = self.all_data["observation"][final_next_index].to(self.device)
#         batch_data["next_legal_action"] = self.all_data["legal_action"][final_next_index].to(self.device)

#         self.sample_id = self.sample_id + self.batch_size

#         return batch_data


# if __name__ == "__main__":
#     # fpath = "hok1v1/datasets/hard_medium"
#     fpath = "hok1v1/datasets/1v1version1/split/"
#     fpath = "hok1v1/datasets/collection_5vs5/"
#     dataset = ParallelLargeDatasets(fpath, batch_size=1000, lstm_steps=4, 
#                             device="cpu", train_step_per_buffer=1000, num_workers=1, 
#                             max_step=500000, dataset_name="test")

#     for i in range(10):
#         b = dataset.next_batch()
#         print(f"get {i} batch, shape", b["done"].shape)

#     dataset= None