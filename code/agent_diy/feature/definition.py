#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_diy.conf.conf import Config
from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
import math

# The create_cls function is used to dynamically create a class.
# The first parameter of the function is the type name, and the remaining parameters are the attributes of the class.
# The default value of the attribute should be set to None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None,
    reward=None,
)


ActData = create_cls(
    "ActData",
    probs=None,
    value=None,
    target=None,
    predict=None,
    action=None,
    prob=None,
)

SampleData = create_cls("SampleData", npdata=None)

RelativeDistance = {
    "RELATIVE_DISTANCE_NONE": 0,
    "VerySmall": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "VeryLarge": 5,
}


RelativeDirection = {
    "East": 1,
    "NorthEast": 2,
    "North": 3,
    "NorthWest": 4,
    "West": 5,
    "SouthWest": 6,
    "South": 7,
    "SouthEast": 8,
}

DirectionAngles = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}

def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


def reward_process(_obs,_extra_info,obs,extra_info,target,target_feature,history_pos,target_in_view,hit_wall,local_memory_map,is_end_pos_found,end_pos,dynamic_o):

    r = 0
    if obs is None:
        r += -0.02*target_feature[-1]
        r += -0.001
        return [r]
    
    if target["type"]=="end":

        
        target_pos_x = target["pos"][0]
        target_pos_z = target["pos"][1]
        pre_x = obs["frame_state"]["heroes"][0]["pos"]["x"]
        pre_z = obs["frame_state"]["heroes"][0]["pos"]["z"]
        cur_x = _obs["frame_state"]["heroes"][0]["pos"]["x"]
        cur_z = _obs["frame_state"]["heroes"][0]["pos"]["z"]
        pre_dist = (pre_x-target_pos_x)**2 + (pre_z-target_pos_z)**2
        cur_dist = (cur_x-target_pos_x)**2 + (cur_z-target_pos_z)**2
        
        if target_in_view:
            r -= 0.05
            if pre_dist>cur_dist:
                r += np.sqrt(abs(pre_dist-cur_dist))*0.025*3*1.5*1.5*3
            else:
                r -= np.sqrt(abs(pre_dist-cur_dist))*0.025*3*1.5*1.5*3

        else:
            r -= 0.01
            if pre_dist>cur_dist:
                r += np.sqrt(abs(pre_dist-cur_dist))*0.025*1.5*3
            else:
                r -= np.sqrt(abs(pre_dist-cur_dist))*0.025*1.5*1.1*3
        
        if _obs["score_info"]["score"] > obs["score_info"]["score"]: # 因为这个分数记录的是总分数
            r += 1

    else:
        
        
        target_pos_x = target["pos"][0]
        target_pos_z = target["pos"][1]
        pre_x = obs["frame_state"]["heroes"][0]["pos"]["x"]
        pre_z = obs["frame_state"]["heroes"][0]["pos"]["z"]
        cur_x = _obs["frame_state"]["heroes"][0]["pos"]["x"]
        cur_z = _obs["frame_state"]["heroes"][0]["pos"]["z"]
        pre_dist = (pre_x-target_pos_x)**2 + (pre_z-target_pos_z)**2
        cur_dist = (cur_x-target_pos_x)**2 + (cur_z-target_pos_z)**2
        
        if target_in_view:
            if pre_dist>cur_dist:
                r += np.sqrt(abs(pre_dist-cur_dist))*0.025*1.5*3
            else:
                r -= np.sqrt(abs(pre_dist-cur_dist))*0.025*1.5*1.5*3

        else:
            
            if pre_dist>cur_dist:
                r += np.sqrt(abs(pre_dist-cur_dist))*0.025*1.5*3
            else:
                r -= np.sqrt(abs(pre_dist-cur_dist))*0.025*1.5*1.1*3

        
        if _obs["score_info"]["score"] > obs["score_info"]["score"] and abs(cur_x-target_pos_x)<=3  and abs(cur_z-target_pos_z)<=3: # 必须是指定目标才行，不能提前吃别的,因为这是后期规划微调
            r += 0.5 

        
        if is_end_pos_found and target["type"]!='end':
            end_pos_x = end_pos[0]
            end_pos_z = end_pos[1]
            pre_dist = (pre_x-end_pos_x)**2 + (pre_z-end_pos_z)**2
            cur_dist = (cur_x-end_pos_x)**2 + (cur_z-end_pos_z)**2
            if abs(end_pos_x-cur_x)<= 3 and abs(end_pos_z-cur_z)<=3:
                if pre_dist>cur_dist:
                    r -= np.sqrt(abs(pre_dist-cur_dist))*0.025
                elif pre_dist == cur_dist:
                    pass
                else:
                    r += np.sqrt(abs(pre_dist-cur_dist))*0.025
        
        

    # step reward
    # 步数奖励
    r += -0.01

    #闪现惩罚

    use_flash = (obs['legal_act'][1] - _obs['legal_act'][1]) == 1 #只有（1-0）才表示使用了闪现
    if use_flash:

        if dynamic_o == 1 and abs(cur_x-42)<=2 and cur_z>=113 and cur_z<=117:
            r -= 0.35
        elif dynamic_o == 2 and abs(cur_x-42)<=2 and cur_z>=78 and cur_z<=81:
            r -= 0.35
        elif dynamic_o == 3 and abs(cur_x-50)<=2 and cur_z>=59 and cur_z<=63:
            r -= 0.35
        elif dynamic_o == 4 and abs(cur_z-60) <= 2 and cur_x>=70 and cur_x<=75:
            r -= 0.35
        elif dynamic_o == 5 and abs(cur_z-41) <= 2 and cur_x>=72 and cur_x<=77:
            r -= 0.35
        elif dynamic_o == 6 and abs(cur_z-42)<=2 and cur_x>=42 and cur_x<=46:
            r -= 0.35
        else:
            r -= 0.6

        
    #撞墙惩罚
    if hit_wall:
        if use_flash:
            r -= 0.3
        
        r-=0.02    #0.02
    
    #局部记忆图
    r -= (np.maximum(
            local_memory_map[3:8,3:8]-0.8, 0)).sum()*0.005  #0.02


    # distance reward
    # 距离奖励
    history_pos_x = history_pos[0]
    history_pos_z = history_pos[1]
    pre_x = obs["frame_state"]["heroes"][0]["pos"]["x"]
    pre_z = obs["frame_state"]["heroes"][0]["pos"]["z"]
    cur_x = _obs["frame_state"]["heroes"][0]["pos"]["x"]
    cur_z = _obs["frame_state"]["heroes"][0]["pos"]["z"]
    pre_dist = (pre_x-history_pos_x)**2 + (pre_z-history_pos_z)**2
    cur_dist = (cur_x-history_pos_x)**2 + (cur_z-history_pos_z)**2

    if pre_dist>cur_dist:
        r += np.sqrt(abs(pre_dist-cur_dist))*0.01  #0,01
    else:
        r -= np.sqrt(abs(pre_dist-cur_dist))*0.01 

    

    return [r]


class SampleManager:
    def __init__(
        self,
        gamma=0.99,
        tdlambda=0.95,
    ):
        self.gamma = Config.GAMMA
        self.tdlambda = Config.TDLAMBDA

        self.feature = []
        self.probs = []
        self.actions = []
        self.reward = []
        self.value = []
        self.adv = []
        self.tdlamret = []
        self.legal_action = []
        self.count = 0
        self.samples = []

    def add(self, feature, legal_action, prob, action, value, reward):
        self.feature.append(feature)
        self.legal_action.append(legal_action)
        self.probs.append(prob)
        self.actions.append(action)
        self.value.append(value)
        self.reward.append(reward)
        self.adv.append(np.zeros_like(value))
        self.tdlamret.append(np.zeros_like(value))
        self.count += 1

    def add_last_reward(self, reward):
        self.reward.append(reward)
        self.value.append(np.zeros_like(reward))

    def update_sample_info(self):
        last_gae = 0
        for i in range(self.count - 1, -1, -1):
            reward = self.reward[i + 1]
            next_val = self.value[i + 1]
            val = self.value[i]
            delta = reward + next_val * self.gamma - val
            last_gae = delta + self.gamma * self.tdlambda * last_gae
            self.adv[i] = last_gae
            self.tdlamret[i] = last_gae + val

    def sample_process(self, feature, legal_action, prob, action, value, reward):
        self.add(feature, legal_action, prob, action, value, reward)

    def process_last_frame(self, reward):
        self.add_last_reward(reward)
        # 发送前的后向传递更新
        # Backward pass updates before sending
        self.update_sample_info()
        self.samples = self._get_game_data()

    def get_game_data(self):
        ret = self.samples
        self.samples = []
        return ret

    def _get_game_data(self):
        feature = np.array(self.feature).transpose()
        probs = np.array(self.probs).transpose()
        actions = np.array(self.actions).transpose()
        reward = np.array(self.reward[:-1]).transpose()
        value = np.array(self.value[:-1]).transpose()
        legal_action = np.array(self.legal_action).transpose()
        adv = np.array(self.adv).transpose()
        tdlamret = np.array(self.tdlamret).transpose()

        data = np.concatenate([feature, reward, value, tdlamret, adv, actions, probs, legal_action]).transpose()

        samples = []
        for i in range(0, self.count):
            samples.append(SampleData(npdata=data[i].astype(np.float32)))

        return samples


@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
