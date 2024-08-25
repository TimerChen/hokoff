import pickle
import os
import traceback
import time
import logging
import json

from collections import deque
import numpy as np
from config.config import Config
from framework.common_log import CommonLogger
from framework.common_log import g_log_time
from framework.common_func import log_time_func

import random
from config.common_config import ModelConfig



# 从文件中加载对象
with open('~/nas/code/sample/hokoff/hok1v1/datasets/06260203/level-0-1/1_0.hdf5', 'rb') as f:
    loaded_camp_index_list = pickle.load(f)
with open('~/nas/code/sample/hokoff/hok1v1/datasets/06260203/level-0-1/1_1.hdf5', 'rb') as f:
    loaded_camp_index_list2 = pickle.load(f)

# 打印加载的对象
print(loaded_camp_index_list["reward"])
print(loaded_camp_index_list2["reward"])