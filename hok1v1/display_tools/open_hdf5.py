import h5py

# 打开HDF5文件
with h5py.File('/NAS2020/Workspaces/DRLGroup/jbhu/code/sample/hokoff/hok1v1/datasets/06260203/level-0-1/1_0.hdf5', 'r') as f:
    # 打印文件中的所有数据集名称
    print("数据集名称：", list(f.keys()))

    # 读取数据集中的数据
    dataset = f['reward']  # 将 'your_dataset_name' 替换为实际的数据集名称
    data1 = dataset[()]  # 读取数据集中的所有数据


with h5py.File('/NAS2020/Workspaces/DRLGroup/jbhu/code/sample/hokoff/hok1v1/datasets/06260203/level-0-1/1_1.hdf5', 'r') as f:
    # 打印文件中的所有数据集名称
    print("数据集名称：", list(f.keys()))

    # 读取数据集中的数据
    dataset = f['reward']  # 将 'your_dataset_name' 替换为实际的数据集名称
    data2 = dataset[()]  # 读取数据集中的所有数据

    # 打印数据



# 从文件中加载对象
# with open('~/nas/code/sample/hokoff/hok1v1/datasets/06260203/level-0-1/1_0.hdf5', 'rb') as f:
#     loaded_camp_index_list = pickle.load(f)
# with open('~/nas/code/sample/hokoff/hok1v1/datasets/06260203/level-0-1/1_1.hdf5', 'rb') as f:
#     loaded_camp_index_list2 = pickle.load(f)

# 打印加载的对象
print(data1.shape)
print(data2.shape)
print(data1)
print(data2)
print(((data1 + data2)==0).sum())