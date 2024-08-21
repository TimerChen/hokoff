import numpy as np
import sys
import re

sys.path.append('.')
import os.path as osp
import datetime
import os
from enum import Enum

# from actor import Actor
# from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import pandas as pd

gc_server_addr = "localhost:23333"
ai_server_ip = '127.0.0.1'


def count_result(root_path, run, train_step, final_test, cpu_num):
    wr_dir = osp.join(root_path, run, "win_rate_details")
    try:
        os.makedirs(wr_dir)
    except FileExistsError:
        pass
    
    all_wr_txt = os.listdir(wr_dir)
    wr_dic = {}  ### find all win rate and save to dict -> model_name(num iterations):[wr] ###
    rwd0_dic = {}
    rwd1_dic = {}
    for wr_txt in all_wr_txt:
        if final_test and int(wr_txt.split('_')[-1].split('.')[0]) >= cpu_num:
            continue
        if wr_txt.startswith('offline_win_rate_'):
            with open(osp.join(wr_dir, wr_txt), 'r') as f:
                lines = f.readlines()
            # if final_test:
                lines = lines[-1:]
            for line in lines:
                try:
                    model_name = re.findall('model_iter: \d+', line)[0].split(' ')[1]  ### which iteration ###
                    wr = float(re.findall('win_rate: \d+\.?\d*', line)[0][10:])
                    print("opopopop")
                    print(wr)
                    reward0 = float(re.findall('reward0: -?\d+\.?\d*', line)[0][9:])
                    reward1 = float(re.findall('reward1: -?\d+\.?\d*', line)[0][9:])
                    print(reward0)
                    if int(model_name) == train_step:  ### catch this train step's win rate ###
                        if model_name in wr_dic.keys():
                            wr_dic[model_name].append(wr)
                        else:
                            wr_dic[model_name] = [wr]
                            
                    if int(model_name) == train_step: 
                        if model_name in rwd0_dic.keys():
                            rwd0_dic[model_name].append(reward0)
                        else:
                            rwd0_dic[model_name] = [reward0]
                            
                    if int(model_name) == train_step: 
                        if model_name in rwd1_dic.keys():
                            rwd1_dic[model_name].append(reward1)
                        else:
                            rwd1_dic[model_name] = [reward1]
                except:
                    print("Parse error for line:", line)
    model_list = []
    win_rate_list = []
    rwd0_list = []
    rwd1_list = []
    print(wr_dic)
    if len(list(wr_dic.keys())) > 0:
        file_path = osp.join(root_path, run, 'eval', 'win_rate_all.txt')
        with open(file_path, 'a') as f:
            for k, v in wr_dic.items():
                print(k, 'win_rate:', sum(v) / len(v), file=f)
                model_list.append(k)
                win_rate_list.append(sum(v) / len(v))
            for k, v in rwd0_dic.items():
                print(k, 'reward0:', sum(v) / len(v), file=f)
                rwd0_list.append(sum(v) / len(v))
            for k, v in rwd1_dic.items():
                print(k, 'reward1:', sum(v) / len(v), file=f)
                rwd1_list.append(sum(v) / len(v))
        print("Win rate saved to", file_path)
    else:
        print("Find no win rate txt!")

    return model_list, win_rate_list


def evaluate(root_path, runs, levels=5, eval_num=1, cpu_num=20, final_test=False, tensorflow_oppo=False, max_steps=500000, dataset_name='level-5-5'):
    print("start evaluate")
    eval_log_len = 0
    print(runs)
    print(max_steps)
    runs = runs.split(';')
    levels = levels.split(';')
    for run_id in range(len(runs)):
        model_step_list = []
        model_pool_path = os.path.join(root_path, runs[run_id])
        if os.path.exists(os.path.join(model_pool_path, 'eval', f'{max_steps}_eval.log')):
            os.remove(os.path.join(model_pool_path, 'eval', f'{max_steps}_eval.log'))
    wait_time_list = [0 for _ in range(len(runs))]
    train_step_list = [-1 for _ in range(len(runs))]
    print(model_step_list)
    # train_step_list = [max_steps]
    tb_summary_list = [SummaryWriter(log_dir=os.path.join(root_path, runs[i], 'eval')) for i in range(len(runs))]
    final_win_rate_list = []
    final_model_num = []
    done_list = [0 for _ in range(len(runs))]
    three_final_win_rate_index = -3
    while True:
        # print(done_list)
        # print(runs)
        if sum(done_list) == len(runs):  ### all evaluations are done ###
            break
        for run_id in range(len(runs)):
            model_step_list = []
            model_pool_path = os.path.join(root_path, runs[run_id])
            for file_name in os.listdir(model_pool_path):
                # print(file_name)
                if '_model' in file_name:
                    model_step = int(file_name.split('_')[0])
                    if model_step <= max_steps:
                        model_step_list.append(model_step)
            model_step_list.sort()
            # print(model_step_list)

            if final_test and model_step_list[-1] < max_steps:  ### if last model isn't max step model, then done ###
                done_list[run_id] = 1
                break

            if len(model_step_list) == 0:  ### current run_id's model not found ###
                continue

            if model_step_list[-1] > train_step_list[run_id]:  ### update train_step_list the latest model ###
                if final_test:
                    train_step_list[run_id] = model_step_list[three_final_win_rate_index]
                    three_final_win_rate_index += 1
                else:
                    train_step_list[run_id] = model_step_list[-1]
            else:
                if train_step_list[run_id] >= max_steps:
                    done_list[run_id] = 1
                continue
            
            print(train_step_list)
            
            
            
            cur_dir_name = os.path.dirname(os.path.realpath(__file__))
            print(os.path.join(model_pool_path, f'{max_steps}_model/model.pth'))
            os.system(
                "sh {}/scripts/start_eval.sh {} {} {} {} {} {} {} {} {} >> {} 2>&1 &".format(
                    cur_dir_name,
                    levels[run_id],
                    eval_num,
                    cpu_num,
                    #train_step_list[run_id],
                    max_steps,
                    # os.path.join(model_pool_path, f'{max_steps}_model/'),
                    runs[run_id],
                    0, #int(runs[run_id].split('_')[-1]),
                    root_path,
                    tensorflow_oppo,
                    dataset_name,
                    os.path.join(model_pool_path, 'eval', f'{max_steps}_eval.log'),
                )
                # "python offline_eval/evaluation.py 
                # --root_path=/NAS2020/Share/jxchen/hokoff/1v1_logs/ 
                # --run_prefix=run_1v1qmix_5v5_10x 
                # --levels=5 
                # --cpu_num=20 
                # --eval_num=100 
                # --final_test=0 
                # --tensorflow_oppo=1 
                # --max_steps=500000 
                # --dataset_name=level-5-5
            )
            start_time = time.time()
            while True:  ### wait for all evaluations are done ###
                if not os.path.exists(os.path.join(model_pool_path, 'eval', f'{max_steps}_eval.log')):
                    continue
                with open(os.path.join(model_pool_path, 'eval', f'{max_steps}_eval.log'), 'r') as f:
                    lines = f.readlines()
                if len(lines) > eval_log_len:
                    if 'All actors are done' in lines[-1]:
                        eval_log_len = len(lines)
                        break
                if time.time() - start_time > 60 * 7 * eval_num:
                    break
                time.sleep(0.1)
            _, win_rate_list = count_result(root_path, runs[run_id], train_step_list[run_id], final_test, cpu_num)
            # with open(os.path.join(model_pool_path, 'eval', 'win_rate_list.log'), 'a') as f:
            #     f.write(f"{max_steps}\n") 
            #     f.write(f"{0 if len(win_rate_list) == 0 else np.mean(win_rate_list)}\n") 
            print(
                f"======evaluate exp {runs[run_id]}: model {train_step_list[run_id]} win rate: {0 if len(win_rate_list)==0 else np.mean(win_rate_list)} total evaluation time: {(time.time()-start_time)/60} min========="
            )
            if len(win_rate_list) > 0:
                tb_summary_list[run_id].add_scalar('eval/win_rate', win_rate_list[0], train_step_list[run_id])
                final_win_rate_list.append(win_rate_list[0])
                final_model_num.append(train_step_list[run_id])
            else:
                final_win_rate_list.append(-1)
                final_model_num.append(train_step_list[run_id])

    if final_test:
        save_win_rate_to_excel(root_path, runs, levels, final_win_rate_list, final_model_num, dataset_name)


def save_win_rate_to_excel(root_path, run_prefix_list, levels_list, win_rate_list, model_num_list, dataset_name):
    if 'general' not in dataset_name:
        excel_dir = os.path.join(root_path, 'final_win_rate.xlsx')
    else:
        excel_dir = os.path.join(root_path, 'final_win_rate_general_{}.xlsx'.format(run_prefix_list[0]))
    data = {}
    data['run_prefix'] = np.array(run_prefix_list * len(win_rate_list))
    data['levels'] = np.array(levels_list * len(win_rate_list))
    data['final_win_rate'] = np.array(win_rate_list)
    data['final_model_num'] = np.array(model_num_list)
    sheet_name = 'test'
    if not os.path.exists(excel_dir):
        print('Create excel file:{}'.format(excel_dir))
        writer = pd.ExcelWriter(excel_dir)
        add_data_df = pd.DataFrame(data, index=list(range(data['run_prefix'].shape[0])))
        add_data_df.to_excel(writer, sheet_name=sheet_name)
        writer.save()
    else:
        print('add data to excel file:{}'.format(excel_dir))

        data_df = pd.read_excel(excel_dir, sheet_name=sheet_name, index_col=False, header=0)
        print(data_df.values)
        data['run_prefix'] = np.concatenate([data_df.values[:, 1], data['run_prefix']], axis=-1)
        data['levels'] = np.concatenate([data_df.values[:, 2], data['levels']], axis=-1)
        data['final_win_rate'] = np.concatenate([data_df.values[:, 3], data['final_win_rate']], axis=-1)
        data['final_model_num'] = np.concatenate([data_df.values[:, 4], data['final_model_num']], axis=-1)
        add_data_df = pd.DataFrame(data, index=list(range(data['run_prefix'].shape[0])))
        writer = pd.ExcelWriter(excel_dir)
        add_data_df.to_excel(writer, sheet_name=sheet_name)
        writer.save()
