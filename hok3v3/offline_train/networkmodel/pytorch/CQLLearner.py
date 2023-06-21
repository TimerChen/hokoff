import torch as th  # in place of tensorflow
import torch.nn as nn  # for builtin modules including Linear, Conv2d, MultiheadAttention, LayerNorm, etc
from torch.nn import ModuleDict  # for layer naming when nn.Sequential is not viable
import numpy as np  # for some basic dimention computation, might be redundent

from math import ceil, floor
from collections import OrderedDict

# typing
from torch import Tensor, LongTensor
from typing import Dict, List, Tuple
from ctypes import Union
from torch.profiler import profile, record_function, ProfilerActivity
from train_eval_config.Config import Config
from train_eval_config.DimConfig import DimConfig
import copy
from networkmodel.pytorch.module.Mixer import QMixer
import time


class CQLLearner:
    def __init__(self, args):
        super(CQLLearner, self).__init__()
        # feature configure parameter
        if 'ind' in args.run_prefix:
            from networkmodel.pytorch.module.IndModel import IndModel as Model

            self.ind = True
        else:
            from networkmodel.pytorch.module.BaseModel import BaseModel as Model

            self.ind = False
        self.args = args
        self.model_name = Config.NETWORK_NAME
        self.lstm_time_steps = args.lstm_time_steps
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.target_embedding_dim = Config.TARGET_EMBEDDING_DIM
        self.hero_data_split_shape = Config.HERO_DATA_SPLIT_SHAPE
        self.hero_seri_vec_split_shape = Config.HERO_SERI_VEC_SPLIT_SHAPE
        self.hero_feature_img_channel = Config.HERO_FEATURE_IMG_CHANNEL
        self.hero_label_size_list = Config.HERO_LABEL_SIZE_LIST
        self.hero_is_reinforce_task_list = Config.HERO_IS_REINFORCE_TASK_LIST

        self.learning_rate = args.lr
        self.var_beta = Config.BETA_START

        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.min_policy = Config.MIN_POLICY
        self.embedding_trainable = False
        self.value_head_num = Config.VALUE_HEAD_NUM

        self.hero_num = 3
        self.hero_data_len = sum(Config.data_shapes[0])
        self.online_net = Model()
        self.target_net = copy.deepcopy(self.online_net)
        self.local_steps = 0
        self.optimizer = th.optim.Adam(params=self.online_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        # self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(target_param.data * self.args.soft_update_tau + param.data * (1 - self.args.soft_update_tau))

    def _build_target(self, data_dict):
        obs = data_dict['observation_s']
        next_obs = data_dict['next_observation_s']
        real_next_obs = th.concat([obs[:, 1:], next_obs], dim=1)  # bs,t,na,d
        # target_local_list,double_target_local_list,_,_ = self.target_net(real_next_obs,only_inference=False)#[[(bs,t,1,13), 25, 42, 42, 39],[13, 25, 42, 42, 39],[13, 25, 42, 42, 39]]
        target_local_list, _, _, _ = self.target_net(
            real_next_obs, only_inference=False
        )  # [[(bs,t,1,13), 25, 42, 42, 39],[13, 25, 42, 42, 39],[13, 25, 42, 42, 39]]

        real_next_legal_action = th.concat([data_dict['legal_action_s'][:, 1:], data_dict['next_legal_action_s']], dim=1)  # bs,t,na,161
        # split_real_next_legal_action = th.split(real_next_legal_action,self.hero_label_size_list[0],dim=-1)#[(bs,t,na,13),(bs,t,na,25),...]
        done = th.unsqueeze(data_dict['done'], dim=-1)  # bs,t,1,1,
        reward = data_dict['reward_s']  # bs,t,na,1,
        masked_chosen_target_local_list = []
        for agent_idx in range(self.hero_num):
            hero_masked_chosen_target_local_list = []
            hero_target_local_list = target_local_list[agent_idx]
            # double_hero_target_local_list = double_target_local_list[agent_idx]
            hero_next_legal_action = real_next_legal_action[:, :, agent_idx : agent_idx + 1]  # bs,t,1,161
            split_hero_next_legal_action = th.split(hero_next_legal_action, self.hero_label_size_list[0], dim=-1)
            for label_index, label_dim in enumerate(self.hero_label_size_list[agent_idx]):
                label_split_hero_next_legal_action = split_hero_next_legal_action[label_index]  # bs,t,1,d

                label_hero_target_local_q = hero_target_local_list[label_index]  # bs,t,1,d
                # double_label_hero_target_local_q = double_hero_target_local_list[label_index]#bs,t,1,d
                masked_label_hero_target_local_q = label_hero_target_local_q * label_split_hero_next_legal_action - 10**10 * (
                    1 - label_split_hero_next_legal_action
                )  # bs,t,1,d
                # double_masked_label_hero_target_local_q = double_label_hero_target_local_q*label_split_hero_next_legal_action-10**10*(1-label_split_hero_next_legal_action)#bs,t,1,d

                # masked_chosen_label_hero_target_local_q = th.max(th.min(masked_label_hero_target_local_q,double_masked_label_hero_target_local_q),dim=-1,keepdim=True)[0]#bs,t,1,1
                masked_chosen_label_hero_target_local_q = th.max(masked_label_hero_target_local_q, dim=-1, keepdim=True)[0]  # bs,t,1,1
                masked_chosen_label_hero_target = (
                    reward[:, :, agent_idx : agent_idx + 1] + self.args.gamma * (1 - done) * masked_chosen_label_hero_target_local_q
                )

                hero_masked_chosen_target_local_list.append(th.detach(masked_chosen_label_hero_target))
            masked_chosen_target_local_list.append(hero_masked_chosen_target_local_list)
        return masked_chosen_target_local_list

    def compute_loss(self, data_dict):
        obs_s = data_dict['observation_s']  # torch.Size([64, 16, 3, 4586])
        legal_action_s = data_dict['legal_action_s']  # torch.Size([64, 16, 3, 161])
        action_s = data_dict['action_s']  # torch.Size([64, 16, 3, 5])
        sub_action_s = data_dict['sub_action_s']  # torch.Size([64, 16, 3, 5])
        local_q_list, _, _, _ = self.online_net(obs_s, only_inference=False)
        target_list = self._build_target(data_dict)

        td_error = 0
        cql_error = 0
        local_q_taken_list = []
        for agent_idx in range(self.hero_num):
            agent_action = action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,5
            agent_sub_action = sub_action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,5
            agent_local_q_list = local_q_list[agent_idx]
            # double_agent_local_q_list = double_local_q_list[agent_idx]
            split_agent_legal_action = th.split(legal_action_s[:, :, agent_idx : agent_idx + 1], self.hero_label_size_list[0], dim=-1)
            for label_index, label_dim in enumerate(self.hero_label_size_list[agent_idx]):
                if th.sum(agent_sub_action[:, :, :, label_index : label_index + 1]) < 1:
                    continue
                chosen_label_agent_local_q = th.gather(
                    agent_local_q_list[label_index], dim=-1, index=agent_action[:, :, :, label_index : label_index + 1].long()
                )  # bs,t,1,1
                # double_chosen_label_agent_local_q = th.gather(double_agent_local_q_list[label_index],dim=-1,index=agent_action[:,:,:,label_index:label_index+1].long())#bs,t,1,1
                if th.isnan(chosen_label_agent_local_q).any():
                    print(label_index, ' q nan')
                    exit(0)
                local_q_taken_list.append(chosen_label_agent_local_q)
                # tmp_td_error_1=th.mean((0.5*(chosen_label_agent_local_q-target_list[agent_idx][label_index].detach())**2)*agent_sub_action[:,:,:,label_index:label_index+1])#/th.sum(agent_sub_action[:,:,:,label_index:label_index+1])
                tmp_td_error_1 = th.mean(
                    (0.5 * (chosen_label_agent_local_q - target_list[agent_idx][label_index].detach()) ** 2)
                    * agent_sub_action[:, :, :, label_index : label_index + 1]
                )  # /th.sum(agent_sub_action[:,:,:,label_index:label_index+1])
                td_error += tmp_td_error_1
                # tmp_td_error_2=(0.5*(double_chosen_label_agent_local_q-target_list[agent_idx][label_index].detach())**2)*agent_sub_action[:,:,:,label_index:label_index+1]
                # td_error+=(th.mean(tmp_td_error_1)+th.mean(tmp_td_error_2))

                masked_agent_label_local_q = agent_local_q_list[label_index] * split_agent_legal_action[label_index] - 10**4 * (
                    1 - split_agent_legal_action[label_index]
                )  # bs,t,1,d
                negative_sampling = th.logsumexp(masked_agent_label_local_q, dim=-1, keepdim=True)  # bs,t,1,1
                cql_error += th.mean(
                    (negative_sampling - chosen_label_agent_local_q) * agent_sub_action[:, :, :, label_index : label_index + 1]
                )  # /th.sum(agent_sub_action[:,:,:,label_index:label_index+1])

                # double_masked_agent_label_local_q = double_agent_local_q_list[label_index]*split_agent_legal_action[label_index]-10**4*(1-split_agent_legal_action[label_index])#bs,t,1,d
                # double_negative_sampling = th.logsumexp(double_masked_agent_label_local_q,dim=-1,keepdim=True)#bs,t,1,1
                # cql_error+=th.mean((double_negative_sampling-double_chosen_label_agent_local_q)*agent_sub_action[:,:,:,label_index:label_index+1])
        loss = (td_error + self.args.cql_alpha * cql_error) / (self.hero_num * len(self.hero_label_size_list[0]))
        return loss, {'td_error': td_error, 'cql_error': cql_error, 'local_q_taken': th.mean(th.stack(local_q_taken_list))}

    def to(self, device):
        self.online_net.to(device)
        self.target_net.to(device)

    # def state_dict(self):
    #     return self.online_net.state_dict()
    # def load_state_dict(self,state_dict):
    #     self.online_net.load_state_dict(state_dict)
    #     self.target_net.load_state_dict(state_dict)
    # def parameters(self):
    #     return self.online_net.parameters()
    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    def step(self, data_dict):
        before_step = time.time()
        self.optimizer.zero_grad()
        with th.cuda.amp.autocast():
            loss, info = self.compute_loss(data_dict)
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(list(self.online_net.parameters()), 10)
        if th.isnan(grad_norm).any():
            print('critic nan!!!!')
            exit(0)
        self.optimizer.step()
        info['step_time'] = time.time() - before_step
        return loss, info

    def save_dict(self):
        save_dict = {
            "network_state_dict": self.online_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        return save_dict

    def load_save_dict(self, save_dict):
        self.online_net.load_state_dict(save_dict['network_state_dict'])
        self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        self.update_target_net()
