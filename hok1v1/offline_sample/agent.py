import random
import h5py
import numpy as np


from config.config import ModelConfig, Config
from framework.predictor.utils import cvt_tensor_to_infer_input, cvt_tensor_to_infer_output
from framework.common_func import log_time
import framework.logging as LOG
import os

import torch


def cvt_infer_list_to_numpy_list(infer_list):
    data_list = [infer.data for infer in infer_list]
    return data_list


class RandomAgent:
    def process(self, feature, legal_action):
        action = [random.randint(0, 2) - 1, random.randint(0, 2) - 1]
        value = [0.0]
        neg_log_pi = [0]
        return action, value, neg_log_pi


class Agent:
    HERO_ID_INDEX_DICT = {
        112: 0,
        121: 1,
        123: 2,
        131: 3,
        132: 4,
        133: 5,
        140: 6,
        141: 7,
        146: 8,
        150: 9,
        154: 10,
        157: 11,
        163: 12,
        169: 13,
        175: 14,
        182: 15,
        193: 16,
        199: 17,
        502: 18,
        513: 19,
    }
    
    def __init__(
        self,
        model_cls,
        keep_latest=False,
        dataset=None,
        backend='pytorch',
    ):
        self.offline_agent = keep_latest
        self.model = model_cls()

        self.backend = backend
        if self.backend == 'pytorch':

            from framework.predictor.predictor.local_torch_predictor import LocalTorchPredictor

            self._predictor = LocalTorchPredictor(self.model)
        else:
            from framework.predictor.predictor.local_predictor import LocalCkptPredictor

            self.graph = self.model.build_infer_graph()
            self._predictor = LocalCkptPredictor(self.graph)

        self.keep_latest = keep_latest

        self.lstm_unit_size = ModelConfig.LSTM_UNIT_SIZE

        self.lstm_hidden = None
        self.lstm_cell = None

        # self.agent_type = "common_ai"
        self.player_id = 0
        self.hero_camp = 0
        self.last_model_path = None
        self.label_size_list = ModelConfig.LABEL_SIZE_LIST
        self.legal_action_size = ModelConfig.LEGAL_ACTION_SIZE_LIST

        self.agent_type = "network"

        if not any(dataset):
            self.save_h5_sample = False
            self.dataset_name = None
            self.dataset = None
            self.tmp_dataset = None
        else:
            self.save_h5_sample = True
            self.dataset_name = dataset
            self.dataset = h5py.File(dataset, "a")
            self.tmp_dataset_name = self.dataset_name[: dataset.rfind('/') + 1] + 'tmp_' + self.dataset_name[dataset.rfind('/') + 1 :]
            if os.path.exists(self.tmp_dataset_name):
                os.remove(self.tmp_dataset_name)
            self.tmp_dataset = h5py.File(dataset, "a")

    def set_game_info(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id

    # reset the agent,agent_type in ["network","common_ai"],if model_path is None,get model from model pool
    def reset(self, agent_type=None, model_path=None):
        # reset lstm input
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])

        if self.backend == 'pytorch':
            self.lstm_hidden = torch.zeros([1, 1, self.lstm_unit_size])
            self.lstm_cell = torch.zeros([1, 1, self.lstm_unit_size])

            self.lstm_step_count = 0
        self.agent_type = "network"

        # for test without model pool
        self._predictor.load_model(model_path)
        self.last_model_path = model_path

        if self.dataset is None:
            self.save_h5_sample = False
        else:
            ### if there exists some data in tmp-dataset, then copy them all to dataset and clear tmp-dataset, else nothing to do ###
            self.save_h5_sample = True
            if len(self.tmp_dataset.keys()) == 0:
                self.dataset.close()
                self.dataset = h5py.File(self.dataset_name, "a")  ### when tmp's key.length == 0 then close and reload ###
                self.tmp_dataset.close()
                self.tmp_dataset = h5py.File(self.tmp_dataset_name, 'a')
            else:
                for key in self.tmp_dataset.keys():  ### copy data from tmp dataset to dataset ###
                    if key not in self.dataset.keys():
                        self.dataset.create_dataset(
                            key,
                            data=np.array(self.tmp_dataset[key]),
                            compression="gzip",
                            maxshape=(None, *(list(self.tmp_dataset[key].shape)[1:])),
                            chunks=True,
                        )
                    else:
                        self.dataset[key].resize(
                            (self.dataset[key].shape[0] + self.tmp_dataset[key].shape[0]), axis=0
                        )  ### keep old data, set new data ###
                        self.dataset[key][-self.tmp_dataset[key].shape[0] :] = np.array(self.tmp_dataset[key])
                self.dataset.close()
                self.dataset = h5py.File(self.dataset_name, "a")
                self.tmp_dataset.close()
                if os.path.exists(self.tmp_dataset_name):  ### clear all tmp dataset, reload new empty tmp dataset ###
                    os.remove(self.tmp_dataset_name)
                self.tmp_dataset = h5py.File(self.tmp_dataset_name, 'a')

    # handle the obs from gamecore, return action result
    @log_time("aiprocess_process")
    def process(self, state_dict, battle=False):

        runtime_id = state_dict["player_id"]
        hero_id = None
        for hero in state_dict["req_pb"].hero_list:
            if hero.runtime_id == runtime_id:
                hero_id = hero.config_id

        if hero_id is None:
            raise Exception("can not find config_id for runtime_id")
        
        hero_id_vec = np.zeros(
            [
                len(self.HERO_ID_INDEX_DICT),
            ],
            dtype=np.float64,
        )
        if self.HERO_ID_INDEX_DICT.get(hero_id) is not None:
            hero_id_vec[self.HERO_ID_INDEX_DICT[hero_id]] = 1
        else:
            LOG.warning("Unknown hero_id for network: %s" % hero_id)
        state_dict["observation"] = np.concatenate(
            (state_dict["observation"], hero_id_vec), axis=0
        )

        feature_vec, legal_action = (
            state_dict["observation"],
            state_dict["legal_action"],
        )

        if self.backend == 'tensorflow':
            pred_ret = self._predict_process(feature_vec, legal_action)  ### prob, value, action, d_action ###
        else:
            pred_ret = self._predict_process_torch(feature_vec, legal_action)
        _, _, action, d_action = pred_ret
        if battle:
            return d_action
        return action, d_action, self._sample_process(state_dict, pred_ret)

    def _update_legal_action(self, original_la, actions):
        target_size = ModelConfig.LABEL_SIZE_LIST[-1]
        top_size = ModelConfig.LABEL_SIZE_LIST[0]
        original_la = np.array(original_la)
        fix_part = original_la[: -target_size * top_size]
        target_la = original_la[-target_size * top_size :]
        target_la = target_la.reshape([top_size, target_size])[actions[0]]
        return np.concatenate([fix_part, target_la], axis=0)

    # build samples from state infos
    def _sample_process(self, state_dict, pred_ret):
        # get is_train
        is_train = False
        req_pb = state_dict["req_pb"]
        for hero in req_pb.hero_list:
            if hero.camp == self.hero_camp:
                is_train = True if hero.hp > 0 else False

        frame_no = req_pb.frame_no
        feature_vec, reward, sub_action_mask = (
            state_dict["observation"],
            state_dict["reward"],
            state_dict["sub_action_mask"],
        )
        done = False
        prob, value, action, d_action = pred_ret  ### prob, value, action, d_action ###

        # legal_action = self._update_legal_action(state_dict["legal_action"], action)
        legal_action = state_dict["legal_action"]  # we need to save all legal actions
        keys = (
            "frame_no",
            "vec_feature",
            "legal_action",
            "action",
            "reward",
            "value",
            "prob",
            "sub_action",
            "lstm_cell",
            "lstm_hidden",
            "done",
            "is_train",
            "all_rewards",
        )
        values = (
            frame_no,
            feature_vec,
            legal_action,
            d_action,
            reward[-1],
            value,
            prob,
            sub_action_mask[d_action[0]],
            self.lstm_cell,
            self.lstm_hidden,
            done,
            is_train,
            reward,
        )
        sample = dict(zip(keys, values))
        self.last_sample = sample

        if self.save_h5_sample:
            self._sample_process_for_saver(sample)
        return sample

    def _get_h5file_keys(self, h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    def _sample_process_for_saver_sp(self, sample_dict, done=False):
        keys = ('done', 'vec_feature')
        keys_in_h5 = self._get_h5file_keys(self.tmp_dataset)
        if not 'done' in keys_in_h5:
            self.tmp_dataset.create_dataset(
                "reward",
                data=[[sample_dict["reward"][-1]]],
                compression="gzip",
                maxshape=(None, 1),
                chunks=True,
            )
            self.tmp_dataset.create_dataset(
                "done",
                data=[[done]],
                compression="gzip",
                maxshape=(None, 1),
                chunks=True,
            )
        else:
            self.tmp_dataset['reward'].resize((self.tmp_dataset['reward'].shape[0] + 1), axis=0)
            self.tmp_dataset['reward'][-1] = [sample_dict["reward"][-1]]

            self.tmp_dataset['done'].resize((self.tmp_dataset['done'].shape[0] + 1), axis=0)
            self.tmp_dataset['done'][-1] = [done]

    def _sample_process_for_saver(self, sample_dict):
        keys = ("frame_no", "vec_feature", "legal_action", "action", "sub_action")
        keys_in_h5 = self._get_h5file_keys(self.tmp_dataset)
        if len(keys_in_h5) == 0:
            self.tmp_dataset.create_dataset(
                "frame_no",
                data=[[sample_dict["frame_no"]]],
                compression="gzip",
                maxshape=(None, 1),
                chunks=True,
            )
            self.tmp_dataset.create_dataset(
                "observation",
                data=[sample_dict["vec_feature"]],
                compression="gzip",
                maxshape=(None, len(sample_dict["vec_feature"])),
                chunks=True,
            )
            self.tmp_dataset.create_dataset(
                "legal_action",
                data=[sample_dict["legal_action"]],
                compression="gzip",
                maxshape=(None, len(sample_dict["legal_action"])),
                chunks=True,
            )
            self.tmp_dataset.create_dataset(
                "action",
                data=[sample_dict["action"]],
                compression="gzip",
                maxshape=(None, len(sample_dict["action"])),
                chunks=True,
            )
            self.tmp_dataset.create_dataset(
                "sub_action",
                data=[sample_dict['sub_action']],
                compression="gzip",
                maxshape=(None, len(sample_dict['sub_action'])),
                chunks=True,
            )
        else:
            for key, value in sample_dict.items():
                if key in keys:
                    key_dataset = key
                    if key_dataset == "vec_feature":
                        key_dataset = "observation"
                    self.tmp_dataset[key_dataset].resize((self.tmp_dataset[key_dataset].shape[0] + 1), axis=0)
                    if isinstance(value, list):
                        self.tmp_dataset[key_dataset][-1] = value
                    else:
                        self.tmp_dataset[key_dataset][-1] = [value]

    # def _sample_process_for_saver_sp(self, sample_dict, done=False):
    #     keys = ('done', 'vec_feature')
    #     keys_in_h5 = self._get_h5file_keys(self.tmp_dataset)
    #     if not 'next_observation' in keys_in_h5:
    #         self.tmp_dataset.create_dataset(
    #             "next_observation",
    #             data=[sample_dict["vec_feature"]],
    #             compression="gzip",
    #             maxshape=(None, len(sample_dict["vec_feature"])),
    #             chunks=True,
    #         )
    #         self.tmp_dataset.create_dataset(
    #             "done",
    #             data=[[done]],
    #             compression="gzip",
    #             maxshape=(None, 1),
    #             chunks=True,
    #         )
    #     else:
    #         self.tmp_dataset['next_observation'].resize((self.tmp_dataset['next_observation'].shape[0] + 1), axis=0)
    #         self.tmp_dataset['next_observation'][-1] = sample_dict["vec_feature"]
    #         self.tmp_dataset['done'].resize((self.tmp_dataset['done'].shape[0] + 1), axis=0)
    #         self.tmp_dataset['done'][-1] = [done]
    # def _sample_process_for_saver(self, sample_dict):
    #     keys = ("frame_no", "vec_feature", "legal_action", "action", "reward")
    #     keys_in_h5 = self._get_h5file_keys(self.tmp_dataset)
    #     if len(keys_in_h5) == 0:
    #         self.tmp_dataset.create_dataset(
    #             "frame_no",
    #             data=[[sample_dict["frame_no"]]],
    #             compression="gzip",
    #             maxshape=(None, 1),
    #             chunks=True,
    #         )
    #         self.tmp_dataset.create_dataset(
    #             "observation",
    #             data=[sample_dict["vec_feature"]],
    #             compression="gzip",
    #             maxshape=(None, len(sample_dict["vec_feature"])),
    #             chunks=True,
    #         )
    #         self.tmp_dataset.create_dataset(
    #             "legal_action",
    #             data=[sample_dict["legal_action"]],
    #             compression="gzip",
    #             maxshape=(None, len(sample_dict["legal_action"])),
    #             chunks=True,
    #         )
    #         self.tmp_dataset.create_dataset(
    #             "action",
    #             data=[sample_dict["action"]],
    #             compression="gzip",
    #             maxshape=(None, len(sample_dict["action"])),
    #             chunks=True,
    #         )
    #         self.tmp_dataset.create_dataset(
    #             "reward",
    #             data=[[sample_dict["reward"]]],
    #             compression="gzip",
    #             maxshape=(None, 1),
    #             chunks=True,
    #         )

    #     else:
    #         for key, value in sample_dict.items():
    #             if key in keys:
    #                 key_dataset = key
    #                 if key_dataset == "vec_feature":
    #                     key_dataset = "observation"
    #                 self.tmp_dataset[key_dataset].resize(
    #                     (self.tmp_dataset[key_dataset].shape[0] + 1), axis=0
    #                 )
    #                 if isinstance(value, list):
    #                     self.tmp_dataset[key_dataset][-1] = value
    #                 else:
    #                     self.tmp_dataset[key_dataset][-1] = [value]

    # given the feature vec and legal_action,return output of the network
    def _predict_process(self, feature, legal_action):
        # put data to input
        input_list = cvt_tensor_to_infer_input(self.model.get_input_tensors())
        input_list[0].set_data(np.array(feature))
        input_list[1].set_data(np.array(legal_action))
        # input_list[2].set_data(label_list)
        input_list[2].set_data(self.lstm_cell)
        input_list[3].set_data(self.lstm_hidden)

        output_list = cvt_tensor_to_infer_output(self.model.get_output_tensors())
        output_list = self._predictor.inference(input_list=input_list, output_list=output_list)
        # cvt output dataxz
        np_output = cvt_infer_list_to_numpy_list(output_list)

        logits, value, self.lstm_cell, self.lstm_hidden = np_output[:4]

        prob, action, d_action = self._sample_masked_action(logits, legal_action)

        return prob, value, action, d_action  # prob: [[ ]], others: all 1D

    def _predict_process_torch(self, feature, legal_action):
        # TODO: add a switch for controlling sample strategy.
        # put data to input
        input_list = []
        input_list.append(np.array(feature))

        torch_inputs = torch.unsqueeze(torch.from_numpy(feature).to(torch.float32), 0)
        self.model.eval()
        with torch.no_grad():
            _, pre_logits = self.model(torch_inputs, True, self.lstm_cell, self.lstm_hidden)

            logits, self.lstm_cell, self.lstm_hidden = pre_logits[:-2], pre_logits[-2], pre_logits[-1]

        logits = torch.cat(logits, dim=-1).numpy()

        prob, action, d_action = self._sample_masked_action(logits, legal_action)

        return prob, 0, action, d_action

    # get final executable actions
    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []
        """
        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits[0], label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],  # [12, 8]
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )

        ### sample-action-sample-action-legal ###
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]  # [12]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])  # [12, 1]
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action  # [12]
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        # target_legal_action = tf.gather(target_legal_action, action_idx, axis=1)
        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        # legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        # prob_list.append(probs)
        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        # Not necessary max clip 1
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        # tmp = tf.exp(tmp - tmp_max)* legal_action + _lsm_const_e
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        # tmp_sum = tf.reduce_sum(tmp, axis=1, keepdims=True)
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        """
        Sample with probability, input probs should be 1D array
        """
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))

    def close(self):
        if self.dataset is not None:
            if len(self.tmp_dataset.keys()) > 0:
                for key in self.tmp_dataset.keys():
                    if key not in self.dataset.keys():
                        self.dataset.create_dataset(
                            key,
                            data=np.array(self.tmp_dataset[key]),
                            compression="gzip",
                            maxshape=(None, *(list(self.tmp_dataset[key].shape)[1:])),
                            chunks=True,
                        )
                    else:
                        self.dataset[key].resize((self.dataset[key].shape[0] + self.tmp_dataset[key].shape[0]), axis=0)
                        self.dataset[key][-self.tmp_dataset[key].shape[0] :] = np.array(self.tmp_dataset[key])
                # self.dataset.close()
                # self.tmp_dataset.close()
                # os.remove(self.tmp_dataset_name)
            self.save_h5_sample = True
            self.dataset.close()
            self.tmp_dataset.close()
            os.remove(self.tmp_dataset_name)
