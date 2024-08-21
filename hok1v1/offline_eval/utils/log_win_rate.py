import os
import os.path as osp
from .ensure_path_exist import ensure_path_exist


def log_win_rate(file_path, actor_id, model_iter, win_rate, reward0, reward1):
    ensure_path_exist(file_path)

    with open(osp.join(file_path, f"offline_win_rate_{actor_id}.txt"), 'a') as f:
        f.write(f"model_iter: {model_iter}\t win_rate: {win_rate}\t reward0: {reward0}\t reward1: {reward1}\n")
