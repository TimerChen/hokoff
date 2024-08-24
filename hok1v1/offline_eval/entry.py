import sys

sys.path.append(".")

import random
import re
import time
import queue
import warnings
import numpy as np
from enum import Enum

from absl import app as absl_app
from absl import flags

from agent import Agent as Agent
from actor import Actor
from baselinemodel.model import Model as BaselineModel
from networkmodel.pytorch.module.OneBaseModel import OneBaseModel as Model
from config.config import Config
from utils.ensure_path_exist import ensure_path_exist
from utils.log_win_rate import log_win_rate
from hok.hok1v1.camp import camp_iterator
import hok.hok1v1.lib.interface as interface
from hok.hok1v1.battle import Battle
from hok.hok1v1.env1v1 import interface_default_config
import os
from hok.hok1v1 import HoK1v1

class ResponceType(Enum):
    NONE = 0
    DEFAULT = 1
    CACHED = 2
    GAMEOVER = 3


FLAGS = flags.FLAGS

flags.DEFINE_integer("actor_id", 0, "actor id")
flags.DEFINE_integer("i", 0, "seed")
flags.DEFINE_integer("max_step", 500, "max step of one round")
flags.DEFINE_integer("train_step", 0, "max step of one round")
flags.DEFINE_string("mem_pool_addr", "localhost:35200", "address of memory pool")
flags.DEFINE_string("model_pool_addr", "localhost:10016", "address of model pool")
flags.DEFINE_string("gamecore_ip", "localhost", "address of gamecore")
flags.DEFINE_integer("thread_num", 1, "thread_num")
flags.DEFINE_boolean("use_lstm", 1, "if use_lstm")

flags.DEFINE_string("agent_models", "", "agent_model_list")
flags.DEFINE_string("dataset_path", "", "dataset save path")
flags.DEFINE_string("levels", "1,1", "levels")

flags.DEFINE_integer("eval_number", 10, "battle number for evaluation")

flags.DEFINE_string("gamecore_path", "~/nas/code/hok_env/hok_env/hok_env_gamecore_20240226/", "installation path of gamecore")
flags.DEFINE_string("game_log_path", "/code/logs/game_log", "log path for game information")
flags.DEFINE_string("offline_log_path", "/code/offline_logs", "log path for offline information")
flags.DEFINE_string("run_prefix", "run5-6_1v1cql", "log path for offline information")
flags.DEFINE_string("dataset_name", 'level-0-0', "dataset name")

flags.DEFINE_string(
    "config_path",
    os.getenv("INTERFACE_CONFIG_PATH", interface_default_config),
    "config file for interface",
)
flags.DEFINE_integer(
    "gamecore_req_timeout",
    30000,
    "millisecond timeout for gamecore to wait reply from server",
)
flags.DEFINE_string(
    "gc_server_addr",
    os.getenv("GAMECORE_SERVER_ADDR", "127.0.0.1"),
    "the gamecore server addr",
)
flags.DEFINE_string(
    "aiserver_ip",
    os.getenv("AI_SERVER_ADDR", "127.0.0.1"),
    "the actor ip",
)

MAP_SIZE = 100
AGENT_NUM = 2
import torch as th

class HoK1v1:
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]
    OBS_SHAPE = [453]
    PLAYER_NUM = 2

    def __init__(
        self,
        runtime_id,
        game_launcher,
        lib_processor,
        addrs,
        eval_mode=False,
        predict_frequency=3,
        aiserver_ip="127.0.0.1",
    ):
        """
        Init the environment

        :param runtime_id:
        :param eval_mode: whether evaluation mode
        :param predict_frequency: predict frequency
        :param aiserver_ip: the ip to access aiserver (default 127.0.0.1)
               if the gamecore server and the actor are not in the same node, you should set aiserver_ip to actor_ip

        """
        self.num_retry = 5
        self.aiserver_ip = aiserver_ip

        self.addrs = addrs
        self.runtime_id = runtime_id
        self.player_ids = {}
        self.camp_list = []
        self.player_list = [None] * self.PLAYER_NUM
        self.player_pos = [
            None,
        ] * self.PLAYER_NUM

        self.is_gameover = True

        self.action_size = self.LABEL_SIZE_LIST

        # init list for split legal action
        self.legal_action_split_size = []
        tmp = 0
        for _, s in enumerate(self.action_size):
            tmp += s
            self.legal_action_split_size.append(tmp)
        self.legal_action_split_size = self.legal_action_split_size

        self.player_masks = {}

        self.cur_frame_no = -1

        self.cur_state = None
        self.cur_req_pb = None
        self.cur_sgame_ids = [None] * self.PLAYER_NUM
        self.eval_mode = eval_mode

        self.repeat_init = [0] * self.PLAYER_NUM

        self.predict_frequency = predict_frequency
        self.request_latency = 1

        self._act_que = []
        self.start_frame = -1

        self.button_names = [
            "None1",
            "None2",
            "Move",
            "Attack",
            "Skill1",
            "Skill2",
            "Skill3",
            "HealSkill",
            "ChosenSkill",
            "Recall",
            "Skill4",
            "EquipSkill",
        ]
        self.category_names = [
            "which_button",
            "move_x",
            "move_z",
            "kill_x",
            "kill_z",
            "target",
        ]
        self.wait_game_max_timeout = 30
        self.game_launcher = game_launcher
        self.lib_processor = lib_processor
        self.servers = []
        print("init complete")

    # util_functions
    def get_random_action(self, info):
        """
        Get random and legal actions based on current states
        """
        actions = []

        shapes = self.action_space()

        split_array = shapes.copy()[:-1]
        for i in range(1, len(split_array)):
            split_array[i] = split_array[i - 1] + split_array[i]

        sub_actions = self.get_subsequent_actions(info)
        for id in range(self.PLAYER_NUM):
            legal_action = np.split(info[id]["legal_action"], split_array)
            legal_action[-1] = legal_action[-1].reshape(
                self.LABEL_SIZE_LIST[0], self.LABEL_SIZE_LIST[-1]
            )

            action = [0] * len(self.LABEL_SIZE_LIST)
            button = random.choice(sub_actions[id]["which_button"])
            action[0] = button
            for category_0 in range(len(self.category_names)):
                if (
                    self.category_names[category_0]
                    == sub_actions[id][self.button_names[button]].keys()
                ):
                    action[category_0] = random.choice(
                        sub_actions[id][self.button_names[button]][category_0]
                    )
                else:
                    if category_0 != len(self.LABEL_SIZE_LIST) - 1:
                        action[category_0] = random.choice(
                            [
                                index
                                for index in range(len(legal_action[category_0]))
                                if legal_action[category_0][index] == 1
                            ]
                        )
                    else:
                        action[category_0] = random.choice(
                            [
                                index
                                for index in range(
                                    len(legal_action[category_0][button])
                                )
                                if legal_action[category_0][button][index] == 1
                            ]
                        )
            actions.append(tuple(action))
        return actions

    def get_noop_action(self, obs=None, info=None):
        """
        get non-op action
        """
        actions = []
        for _ in range(self.PLAYER_NUM):
            actions.append([0] * len(self.action_space()))
        return actions

    def get_subsequent_actions(self, info):
        """
        get the legal actions, first key is the button, second key is the botton, how and who
        """
        shapes = self.action_space()

        split_array = shapes.copy()[:-1]
        for i in range(1, len(split_array)):
            split_array[i] = split_array[i - 1] + split_array[i]

        subsequent_actions = [None] * self.PLAYER_NUM
        for id in range(self.PLAYER_NUM):
            if not self.need_predict(id):
                continue
            subsequent_actions[id] = dict()
            # first, make keys as the which buttion
            sub_action = info[id]["sub_action_mask"]

            legal_action = np.split(info[id]["legal_action"], split_array)
            legal_action[-1] = legal_action[-1].reshape(
                self.LABEL_SIZE_LIST[0], self.LABEL_SIZE_LIST[-1]
            )

            allowed_button = [
                index
                for index in range(len(legal_action[0]))
                if legal_action[0][index] == 1
            ]
            subsequent_actions[id][self.category_names[0]] = allowed_button
            subsequent_actions[id]["legal_action"] = legal_action
            for button in allowed_button:
                button_name = self.button_names[button]
                allowed_categories = [
                    index
                    for index in range(len(sub_action[button]))
                    if sub_action[button][index] == 1
                ]
                subsequent_actions[id][button_name] = dict()
                for category in allowed_categories:
                    category_name = self.category_names[category]
                    if category == 0:
                        continue
                    elif category == len(self.LABEL_SIZE_LIST) - 1:
                        subsequent_actions[id][button_name][category_name] = [
                            index
                            for index in range(len(legal_action[category][button]))
                            if legal_action[category][button][index] == 1
                        ]
                    else:
                        subsequent_actions[id][button_name][category_name] = [
                            index
                            for index in range(len(legal_action[category]))
                            if legal_action[category][index] == 1
                        ]

        return subsequent_actions

    def action_space(self):
        """
        the action space of an agent in the environment
        """
        return self.LABEL_SIZE_LIST

    def obs_space(self):
        """
        the observation space of the agent in the environment
        """
        return self.OBS_SHAPE

    def _state2ret(self, states):
        obs = [None] * self.PLAYER_NUM
        reward = [0.0] * self.PLAYER_NUM
        done = [False] * self.PLAYER_NUM
        info = states
        for i in range(self.PLAYER_NUM):
            if not self.need_predict(i):
                continue
            obs[i] = states[i]["observation"]
            temp_reward = states[i]["reward"][:6] + states[i]["reward"][7:]
            reward[i] = temp_reward
            done[i] = states[i]["done"]

        return obs, reward, done, info

    def _split_legal_action(self, la, button):
        tmp = np.split(la, self.legal_action_split_size[:-1])
        tmp[-1] = tmp[-1].reshape(-1, self.LABEL_SIZE_LIST[-1])[button]
        return tmp

    def _check_action(self, actions):
        # check whether the actions are legal
        for i, act in enumerate(actions):
            if not self.need_predict(i):
                continue
            legal = self.cur_state[i]["legal_action"]
            legal = self._split_legal_action(legal, act[0])
            sub = self.cur_state[i]["sub_action_mask"][int(act[0])]

            for j in range(6):
                a = int(act[j])
                if (a < 0 or a >= len(legal[j])) or legal[j][a] == 0:
                    warnings.warn(
                        "Agent[{}] is passed with an illegal action {} No.{}:[{}], legal: {}, all: {}, sub: {}".format(
                            i, self.category_names[j], j, a, legal[j], act, sub
                        )
                    )

    def step(self, actions):
        """
        pass actions to environment
        """
        self._check_action(actions)
        self._step_action(actions)
        state, req_pb = self._step_feature()
        # print(aaa)
        self._update_gameover(state, req_pb)
        # print(bbb)
        # return state, req_pb
        self.cur_req_pb = req_pb
        self.cur_state = state
        return self._state2ret(state)

    def _state_tuple2np(self, states):
        states = list(states)
        for _, state in enumerate(states):
            if state is None:
                continue
            for k in state:
                if isinstance(state[k], tuple) and k in ["legal_action", "observation"]:
                    state[k] = np.array(state[k])
                if isinstance(state[k], dict) and k in ["sub_action_mask"]:
                    # tmp_tuple = {}
                    for i in state[k]:
                        state[k][i] = np.array(state[k][i])

        return states

    def _update_gameover(self, state, req_pb):
        for i in range(self.PLAYER_NUM):
            print("loiiloi")
            if not self.need_predict(i):
                continue
            state[i]["done"] = state[i]["done"] or req_pb[i].gameover

    def _format_actions(self, actions):
        # check whether the actions are within defined range, and format into gamecore actions
        rp_actions = []
        for i, action in enumerate(actions):
            if self.player_masks[i]:
                action = [0.0] * 6
            # formulation check
            if isinstance(action, (tuple, list)):
                if not len(action) == 6:
                    assert False, "action[{}] length incorrect: {}, but expect 6."
                action = np.array(action)
            elif isinstance(action, np.ndarray):
                if not (len(action.shape) == 1 and action.shape[0] == 6):
                    assert (
                        False
                    ), "action[{}] shape incorrect: {}, but expect [6].".format(
                        i, action.shape
                    )
            else:
                assert False, "invalid action[{}] type of {}".format(i, type(action))
            old_action = action
            action = []
            for j, act in enumerate(old_action):
                assert (
                    0 <= act < self.action_size[j]
                ), "Action[{}] {}: {} not in [0,{})".format(
                    i, j, act, self.action_size[j]
                )
                action.append((act,))
            action = tuple(action)
            rp_act = (
                (
                    (0,) * 12,
                    (0,) * 16,
                    (0,) * 16,
                    (0,) * 16,
                    (0,) * 16,
                    (0,) * 8,
                )
                + action
                + ((0,),)
            )
            rp_actions.append(rp_act)
        return tuple(rp_actions)

    def _step_action(self, actions):
        # 6 actions: top, move_x, move_z, skill_x, skill_z, target
        # msg format to RP:
        #   logp: 0(11) 1(15) 2(15) 3(15) 4(15), target(7)
        #   sample: 0, 1, 2, 3, 4, target
        #   value: [1]
        rp_actions = self._format_actions(actions)
        s_msgs = [None] * self.PLAYER_NUM
        for id in range(self.PLAYER_NUM):
            if not self.need_predict(id):
                # silently skip if is common ai
                continue

            ret_code, resp_id = self.lib_processor.ResultProcess(
                rp_actions[id : id + 1], self.cur_sgame_ids[id]
            )

            assert (
                ret_code == interface.PROCESS_ACTION_SUCCESS
            ), "process action failed: {}".format(ret_code)

            s_msgs[id] = (ResponceType.CACHED, resp_id)
        self._send_all(s_msgs)
        # self._send(ret[1], id)

    def need_predict(self, id):
        """
        Whether the player needs passed action
        """
        return not self.player_masks[id]

    def _init_hero_info(self, pb, id):
        if self.player_list[id] is not None:
            return
        if 0 < len(pb.command_info_list):
            # set current player
            for cmd in pb.command_info_list:
                print("cmd.player_id", cmd.player_id)
                self.player_list[id] = cmd.player_id

            # get camp info
            for hero_state in pb.hero_list:
                if hero_state.runtime_id == self.player_list[id]:
                    self.player_ids.setdefault(hero_state.camp, []).append(
                        hero_state.runtime_id
                    )
                    self.camp_list.append(hero_state.camp)
                    self.player_camp[hero_state.runtime_id] = hero_state.camp

    def _recv_all(self):
        print("start recv")
        pbs = []
        parse_ret = []
        for id in range(self.PLAYER_NUM):
            if not self.need_predict(id):
                pbs.append(None)
                parse_ret.append(None)
                continue
            # import pdb;pdb.set_trace();
            parse_state, sgame_id = self.lib_processor.RecvAIFrameState(self.addrs[id])
            self.cur_sgame_ids[id] = sgame_id
            assert (
                parse_state == interface.PARSE_CONTINUE
                or parse_state == interface.PARSE_NONE_ACTION
            ), ("step failed: %s" % parse_state)

            req_pb = None
            if parse_state == interface.PARSE_CONTINUE:
                req_pb = self.lib_processor.GetAIFrameState(sgame_id)
                assert req_pb is not None, "GetAIFrameState failed"

            if self._act_que[id].empty():
                # first frame!
                # update player info with info
                self._init_hero_info(req_pb, id)
                for _ in range(self.request_latency):
                    self._act_que[id].put((ResponceType.NONE, -1))

            smsg = self._act_que[id].get()

            self._send(smsg, id)
            pbs.append(req_pb)
            parse_ret.append((parse_state, sgame_id))
            print("end recv")
        return parse_ret, pbs

    def _send_all(self, msgs, ignore_none=False):
        # msgs: list of (ResponceType, param)
        # only put rsp_msg in buffer
        for id in range(self.PLAYER_NUM):
            if not self.need_predict(id):
                continue
            if msgs[id] is None:
                if ignore_none:
                    # self._send(self._get_rsp_pb(id).SerializeToString(), id)
                    self._act_que[id].put((ResponceType.NONE, -1))
                else:
                    assert False, "send None at {}".format(id)
            else:
                self._act_que[id].put(msgs[id])

    def _step_feature(self, first_frame=False):
        while True:
            ret_flag = [False] * self.PLAYER_NUM
            ret_num = 0
            real_num = 0
            states = [None] * self.PLAYER_NUM

            r_msgs, req_pbs = self._recv_all()
            print("_step_feature: ", req_pbs[0].frame_no)
            s_msgs = [None] * self.PLAYER_NUM
            for id in range(self.PLAYER_NUM):
                print("PPPPPPPPPPPP")
                if not self.need_predict(id):
                    # silently skip if is common ai
                    continue
                real_num += 1
                req_pb = req_pbs[id]
                self.cur_frame_no = req_pb.frame_no
                if first_frame:
                    self.start_frame = self.cur_frame_no
                print(1)
                if req_pb.gameover:
                    print("game is over!")
                    self.is_gameover = True

                if (not first_frame) and (
                    self.cur_frame_no - self.start_frame
                ) % self.predict_frequency > 0:
                    print(2)
                    # skip this non-predict frame
                    # if gameover, skip c++ process, so checking gameover with py code is necessary.
                    if not self.is_gameover:
                        s_msgs[id] = (ResponceType.NONE, -1)
                        continue
                print(3)
                parse_state, sgame_id = r_msgs[id]
                ret = self.lib_processor.FeatureProcess(parse_state, sgame_id)
                print(4)
                # Failed, return no action
                if ret[0] == 0:
                    print(5)
                    # LOG.error("step failed: {}".format(ret[1]))
                    assert False, "step failed: {}".format(ret[1])
                if ret[0] == 1:
                    print(6)
                    # directly receive msg again!
                    assert False, "Parsing gameover information, receive msg again!"
                    # LOG.error("Parsing gameover information, receive msg again!")
                elif ret[0] == 2:
                    # SEND_CCD_ONE_HERO, get normal feature vector, break
                    state = self._state_tuple2np(ret[1:])[0]
                    if req_pb.gameover:
                        print(
                            "gameover at frameno {} of {}!".format(req_pb.frame_no, id)
                        )
                    # LOG.debug("Parsing normal feature of len({}).".format(len(states)))
                    ret_num += 1
                    state["req_pb"] = req_pb
                    state["sgame_id"] = sgame_id
                    states[id] = state
                    ret_flag[id] = True
                    s_msgs[id] = None
                    # return states, req_pb
                elif ret[0] == 3 or ret[0] == 4 or ret[0] == 5:
                    print(7)
                    # SEND_CCD_FIVE_HERO
                    self.repeat_init[id] = 0
                    if ret[0] == 3:
                        s_msgs[id] = (ResponceType.DEFAULT, -1)
                    elif ret[0] == 4:
                        s_msgs[id] = (ResponceType.NONE, -1)
                    elif ret[0] == 5:
                        s_msgs[id] = (ResponceType.CACHED, int(ret[1]))
            print(8)
            if first_frame:
                if ret_num < real_num:
                    self._send_all(s_msgs, ignore_none=True)
                    continue
            elif (self.cur_frame_no - self.start_frame) % self.predict_frequency > 0:
                if not self.is_gameover:
                    self._send_all(s_msgs)
                    continue
            print(9)
            return states, req_pbs

    def close_game(self):
        """
        close game by sending signals to gamecore
        """
        while not self.is_gameover:
            print("game not end, send close game at first", self.cur_frame_no)
            for i in range(self.PLAYER_NUM):

                # silently skip if is common ai
                if not self.need_predict(i):
                    continue

                parse_state, sgame_id = self.lib_processor.RecvAIFrameState(
                    self.addrs[i]
                )
                if parse_state != interface.PARSE_CONTINUE:
                    continue
                req_pb = self.lib_processor.GetAIFrameState(sgame_id)
                if req_pb.gameover:
                    self.is_gameover = True

                self._gameover(i)

        # wait game over
        self.game_launcher.wait_battle(self.runtime_id, self.wait_game_max_timeout)

        # force close
        self.game_launcher.stop_battle(self.runtime_id)

    def reset(
        self,
        camp_hero_list=None,
        use_common_ai=None,
        eval=None,
    ):
        """
        resetting the environment

        :param use_common_ai: whether to use common_ai
        :param eval: whether is evaluation mode
        :return: obs, reward, done, info
        """
        if use_common_ai is None:
            use_common_ai = [False] * self.PLAYER_NUM
        if eval is not None:
            self.eval_mode = eval
        print(1)
        # setup aiserver
        for addr in self.addrs:
            print(2)
            if not self.lib_processor.server_manager.Has(addr):
                print(3)
                for j in range(self.num_retry):
                    print(4)
                    zmq_server = None
                    try:
                        print(5)
                        zmq_server = self.lib_processor.server_manager.Add(addr)
                        if not zmq_server:
                            raise Exception("Address already exists: {}".format(addr))
                        rc = zmq_server.Reset(addr)
                        print(6)
                        if rc < 0:
                            print(rc)
                            raise Exception(
                                "zmq_server.Reset failed: %s, retry %d/%d"
                                % (rc, j + 1, self.num_retry)
                            )
                        break
                    except Exception:
                        print(7)
                        if zmq_server:
                            zmq_server.Close()
                        self.lib_processor.server_manager.Delete(addr)
                        time.sleep(1)
                        if j == self.num_retry - 1:
                            raise
        print(2)
        # reset infos
        sgame_ids = []
        if self.cur_sgame_ids:
            for sgame_id in self.cur_sgame_ids:
                if sgame_id:
                    sgame_ids.append(sgame_id)
        self.lib_processor.Reset(self.eval_mode, sgame_ids)
        print("a")

        self.last_predict_frame = 0
        self.player_ids = {}
        self.player_camp = {}
        self.camp_list = []
        self.reward = []
        self.camp_hero_list = camp_hero_list
        self.player_list = [None] * self.PLAYER_NUM
        self.repeat_init = [0] * self.PLAYER_NUM
        self.is_gameover = False
        print("b")

        self.player_masks = use_common_ai.copy()
        print("c")

        self.game_launcher.start_battle(
            self.runtime_id, self._get_server(use_common_ai), camp_hero_list
        )
        print("d")
        self._act_que = [queue.Queue() for _ in range(self.PLAYER_NUM)]
        
        states, req_pbs = self._step_feature(True)
        print("nmnmn")
        self._update_gameover(states, req_pbs)
        # return state, req_pb
        self.cur_state = states
        self.cur_req_pb = req_pbs
        return self._state2ret(states)

    def _get_server(self, use_common_ai):
        servers = []
        for i, addr in enumerate(self.addrs):
            # addr == "tcp://0.0.0.0:{}"
            servers.append(
                None
                if use_common_ai[i]
                else (self.aiserver_ip, int(addr.split(":")[-1]))
            )
        return servers

    def _send(self, msg, id):
        send_type, msg_id = msg
        ret = None
        if send_type == ResponceType.GAMEOVER:
            ret = self.lib_processor.SendGameoverResp(
                self.addrs[id], self.cur_sgame_ids[id]
            )
        elif send_type == ResponceType.DEFAULT:
            ret = self.lib_processor.SendDefaultResp(self.addrs[id])
        elif send_type == ResponceType.NONE:
            ret = self.lib_processor.SendNoneResp(
                self.addrs[id], self.cur_sgame_ids[id]
            )
        elif send_type == ResponceType.CACHED:
            ret = self.lib_processor.SendResp(
                self.addrs[id], self.cur_sgame_ids[id], msg_id
            )
        else:
            assert False, "Unknown ResponceType: %s" % send_type

        assert ret == interface.SEND_SUCCESS, "Send resp failed: %s" % ret

    def _gameover(self, id):
        self._send((ResponceType.GAMEOVER, -1), id)



#  gamecore as lib
def gc_as_lib(argv):
    # from hok.hok1v1 import HoK1v1

    thread_id = 0
    actor_id = FLAGS.thread_num * FLAGS.i + thread_id
    seed = FLAGS.actor_id
    th.set_num_threads(FLAGS.thread_num)
    agents = []
    game_id_init = "None"
    main_agent = random.randint(0, 1)
    
    # if not os.path.exists('../logs/game_log'):
    #     os.makedirs('../logs/game_log')

    eval_number = FLAGS.eval_number
    if FLAGS.levels.split(',')[1] == -1:
        enemy_model = 'common_ai'
        enemy_level = -1
    else:
        enemy_level = FLAGS.levels.split(',')[1]
        enemy_model = '../baselines/tensorflow/level-' + FLAGS.levels.split(',')[1] + '/algorithms/checkpoint'

    load_models = [FLAGS.agent_models, enemy_model]

    print('Enemy level:', enemy_level)
    print(load_models)
    for i, m in enumerate(load_models):
        if m == "common_ai":
            load_models[i] = None
    eval_mode = eval_number > 0

    import os
    gc_server_addr = os.getenv("GAMECORE_SERVER_ADDR")
    ai_server_addr = os.getenv("AI_SERVER_ADDR")
    if gc_server_addr is None or len(gc_server_addr) == 0 or "127.0.0.1" in gc_server_addr:
        # local gc server
        gc_server_addr = "127.0.0.1:23333" #mark!!!
        ai_server_addr = "127.0.0.1"
        remote_mode = 1
    else:
        # remote gc server
        remote_mode = 2
    remote_param = {
        "remote_mode": remote_mode,
        "gc_server_addr": gc_server_addr,
        "ai_server_addr": ai_server_addr,
    }
    gc_mode = os.getenv("GC_MODE")
    if gc_mode == "local":
        remote_param = None
    
    print("load config.dat: ", FLAGS.config_path)
    lib_processor = interface.Interface()
    lib_processor.Init(FLAGS.config_path)

    game_launcher = Battle(
        server_addr=gc_server_addr,
        gamecore_req_timeout=FLAGS.gamecore_req_timeout,
    )

    addrs = []
    for i in range(AGENT_NUM):
        print("id_____________--")
        print(actor_id)
        addrs.append("tcp://0.0.0.0:{}".format(35300 + actor_id * AGENT_NUM + i + int(FLAGS.train_step / 500)))
    
    print(len(addrs))
    print(addrs[0])
    print(addrs[1])
    env = HoK1v1(
        "actor-1v1-{}".format(actor_id),
        game_launcher,
        lib_processor,
        addrs,
        aiserver_ip=FLAGS.aiserver_ip,
    )

    if any(FLAGS.dataset_path):
        dataset_path = [FLAGS.dataset_path[:-5] + "_" + str(i) + ".hdf5" for i in range(AGENT_NUM)]
        for path in dataset_path:
            ensure_path_exist(path[:-8])
    else:
        dataset_path = ["" for i in range(AGENT_NUM)]
    agents.append(Agent(Model, keep_latest=1, dataset=dataset_path, backend='pytorch'))
    agents.append(Agent(BaselineModel, keep_latest=0, dataset=dataset_path, backend='tensorflow'))

    actor = Actor(id=seed, agents=agents, gpu_ip=FLAGS.mem_pool_addr.split(":")[0])
    actor.set_env(env)
    
    import os 
    cur_dir_name = os.path.dirname(os.path.realpath(__file__))
    cur_dir_dir_name = os.path.dirname(cur_dir_name)
    if 'sub_task' in FLAGS.dataset_name:
        actor.sub_task = True
    if 'multi_hero_oppo' in FLAGS.dataset_name:
        env_config_path = '{}/offline_eval/hero_config/multi_hero_oppo_config/hero_config_1.json,{}/offline_eval/hero_config/multi_hero_oppo_config/hero_config_2.json'.format(cur_dir_name, cur_dir_name)
    elif 'multi_oppo' in FLAGS.dataset_name:
        env_config_path = '{}/hero_config/multi_oppo_config/hero_config_1.json,{}/hero_config/multi_oppo_config/hero_config_2.json'.format(cur_dir_name, cur_dir_name)
    elif 'multi_hero' in FLAGS.dataset_name:
        env_config_path = '{}/hero_config/multi_hero_config/hero_config_1.json,{}/hero_config/multi_hero_config/hero_config_2.json'.format(cur_dir_name, cur_dir_name)
    elif 'hard_expert' in FLAGS.dataset_name:
        env_config_path = '{},{}'.format(
            '{}/baselines/tensorflow/level-7/hero_config.json'.format(cur_dir_dir_name), load_models[1][: -len('algorithms/checkpoint')] + 'hero_config.json'
        )
    else:
        env_config_path = '{},{}'.format(
            load_models[1][: -len('algorithms/checkpoint')] + 'hero_config.json', load_models[1][: -len('algorithms/checkpoint')] + 'hero_config.json'
        )

    offline_win_rate, reward0, reward1 = actor.run(eval_mode=eval_mode, eval_number=eval_number, load_models=load_models, env_config_path=env_config_path,hero_levels=FLAGS.levels.split(','))
    print(10000)
    print(offline_win_rate)
    print(reward0)
    print(reward1)
    log_win_rate(os.path.join(FLAGS.offline_log_path, FLAGS.run_prefix, 'win_rate_details'), actor_id, FLAGS.train_step, offline_win_rate, reward0, reward1)
    ############changed


if __name__ == "__main__":
    absl_app.run(gc_as_lib)
