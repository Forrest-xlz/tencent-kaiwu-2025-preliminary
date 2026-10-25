#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
from agent_diy.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process
import agent_diy.feature.constants as const

def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)

class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 16
        self.reset()
        self._obs = None
        self._extra_info = None
        # 首先一样的初始化
        self.global_memory_map = np.zeros((128,128), dtype=np.float32)
        self.local_memory_map = np.zeros((11,11), dtype=np.float32)
        self.action = None


    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []
        self.bad_move_ids = set()
        self.is_flashed = True
        self.target = None
        self.obs = None
        self.extra_info = None
        self._obs = None
        self._extra_info = None
        self.target_in_view = False
        self.action = None
        self.global_memory_map = np.zeros((128,128), dtype=np.float32)
        self.local_memory_map = np.zeros((11,11), dtype=np.float32)
        self.hit_wall = None
        self.ka = False
        self.last_target = None
        self.score_info = None
        self.treasure_map = {
            "80,50":1,
            "64,20":1,
            "109,52":1,
            "46,55":1,
            "20,84":1,
            "26,116":1,
            "66,97":1,
            "101,28":1,
            "85,113":1,
            "38,55":1,
        }
        self.dir = None
        self.youshang = None
        self.dynamic_o = 0
        self.zuoxia = None
        self.zhongxia = None
        self.youzhong = None
        self.end_mhd_dist = None
        self.zuoxia_add = None

     



    def _get_pos_feature(self, found, cur_pos, target_pos):
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, 128, -128)
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128),
            ),
        )
        return feature
    

    def get_target(self):

        
        
        
            min_dist = 1e9
            target_pos = None
            config_id = None
            #先确定最近的目标是什么
            for i in range(1,14):
                pos = self.target_dist[i]
                if self.organs_states[i]!=0:
             
                    dist = np.linalg.norm([(self.cur_pos[0]-pos[0]),(self.cur_pos[1]-pos[1])])
                    if dist<min_dist:
                        min_dist = dist
                        target_pos = pos
                        config_id = i
            
            # 路线规划

            # 右区
            if self.last_target is not None and self.last_target['pos'] == (101,28) and self.treasure_map["109,52"] == 0:
                
                return {"type":"pot","pos":(100,52),"config_id":0}
            
            if self.last_target is not None and self.last_target['pos'] == (100,52):
                
                return {"type":"pot","pos":(80,50),"config_id":0}
            
            if self.last_target is not None and self.last_target["pos"] == (101,28) and self.treasure_map["109,52"] == 1:
                
                return {"type":"pot","pos":(109,52),"config_id":0}
            if self.last_target is not None and self.last_target["pos"] == (109,52) and self.treasure_map["101,28"] == 1:
                
                return {"type":"pot","pos":(101,28),"config_id":0}
            
            # 上区
            if self.last_target is not None and self.last_target["pos"] == (46,55) and target_pos == (20,84):
                
                return {"type":"pot","pos":(66,97),"config_id":0}
            
            if self.last_target is not None and self.last_target['pos'] == (20,84) and sum(self.organs_states[1:14]) == 0 and self.end_pos[0]<=40  and self.end_pos[1]>=75 and self.is_end_pos_found:   
                
                return {"type":"end","pos":self.target_dist[14],"config_id":14} #这个位置必须在下一行代码前面
            
            if self.last_target is not None and self.last_target['pos'] == (20,84) and self.treasure_map["26,116"] == 0 and self.treasure_map["66,97"] == 0:
                
                return {"type":"buff","pos":(57,80),"config_id":0}
            
            
            if self.last_target is not None and self.last_target['pos'] == (57,80) and self.treasure_map["26,116"] == 0 and self.treasure_map["66,97"] == 0:
                
                if sum(self.organs_states[1:14]) == 0 and (not self.end_far):
                    return {"type":"buff","pos":(61,75),"config_id":0}
                else:
                    return {"type":"buff","pos":self.target_dist[0],"config_id":0}
                
            
            if self.last_target is not None and self.last_target['pos'] == (20,84) and self.treasure_map["26,116"] == 1:
            
                return {"type":"treasure","pos":(26,116),"config_id":0}  
            
            if self.last_target is not None and self.last_target['pos'] == (20,84) and self.treasure_map["66,97"] == 1:
            
                return {"type":"treasure","pos":(66,97),"config_id":0}  
            
            if self.last_target is not None and self.last_target["pos"] == (66,97) and self.youshang == 1 and self.treasure_map["85,113"] == 1 and not self.is_end_pos_found:
                return {"type":"treasure","pos":(85,113),"config_id":0}
            
            if self.last_target is not None and self.last_target["pos"] == (85,113):
                return {"type":"treasure","pos":(66,97),"config_id":0}
            

            
             
            
            # 中区
            if self.last_target is not None and self.last_target["pos"] == (46,55) and self.treasure_map["64,20"] == 1:
                self.dir = "left"

                return {"type":"treasure","pos":(64,20),"config_id":0}

            if self.last_target is not None and self.last_target["pos"] == (80,50) and self.treasure_map["64,20"] == 1 and self.treasure_map["109,52"] == 0 and self.treasure_map["101,28"] == 0:
                self.dir = "right"
      
                return {"type":"treasure","pos":(64,20),"config_id":0}

            if self.last_target is not None and self.last_target["pos"] == (64,20) and self.dir == "left":
              
                return {"type":"treasure","pos":(80,50),"config_id":0}
            
            if self.last_target is not None and self.last_target["pos"] == (64,20) and self.dir == "right":
          
                return {"type":"treasure","pos":(46,26),"config_id":0}
            
            if self.last_target is not None and self.last_target["pos"] == (46,26) and self.dir == "right":
          
                return {"type":"treasure","pos":(46,55),"config_id":0}

            
            if self.last_target is not None and self.last_target["pos"] == (64,20) and self.dir == None:

                return {"type":"treasure","pos":(46,26),"config_id":0}
            
            if self.last_target is not None and self.last_target["pos"] == (46,26) and self.dir == None:

                return {"type":"treasure","pos":(46,55),"config_id":0}
            
            if self.last_target is not None and self.last_target["pos"] == (46,55) and self.zuoxia == 6 and self.is_end_pos_found == False and self.treasure_map["38,55"] == 1:

                return {"type":"treasure","pos":(38,55),"config_id":0}
            
            if self.last_target is not None and self.last_target["pos"] == (38,55):
                return {"type":"treasure","pos":(46,55),"config_id":0}



            if self._cur_z<=39 and self._cur_x>=38 and self._cur_x<=85 and self.treasure_map["64,20"] == 1: #新增

                return {"type":"treasure","pos":(64,20),"config_id":0}
            
            if self.last_target is not None and self.last_target["pos"] == (46,55) and self.treasure_map["64,20"] == 0 and self.treasure_map["80,50"] == 1:
                return {"type":"treasure","pos":(57,68),"config_id":0}
            
            if self.last_target is not None and self.last_target["pos"] == (80,50) and self.treasure_map["64,20"] == 0 and self.treasure_map["109,52"] == 0 and self.treasure_map["101,28"] == 0 and (self.treasure_map["46,55"] == 1 or (self.zuoxia in [5,6,7])):
                return {"type":"treasure","pos":(57,68),"config_id":0}
            
            
            
            
            
            # 额外

            if sum(self.organs_states[1:14]) != 0 and self._cur_x<=36 and self._cur_z<=63 and ((self.last_target is not None and self.last_target["pos"] != (34,56) and self.last_target["type"]!='end') or self.last_target is None):  #起点在左下角
                return {"type":"treasure","pos":(34,56),"config_id":0}
            
            if sum(self.organs_states[1:14]) == 0 and self.target_dist[14][0]>85 and self.last_target is not None and self.last_target["pos"] != (90,50) and self.last_target["type"] != 'end'and (self.youshang not in [2,3] or (self.youshang == 1 and self.is_end_pos_found == False )): #处理右上终点问题
                return {"type":"treasure","pos":(90,50),"config_id":0}
            
            if sum(self.organs_states[1:14]) == 0 and self.last_target is not None and self.last_target["pos"] != (66,97) and self.last_target["type"] != 'end'and (self.youshang in [2,3] or (self.youshang == 1 and self.is_end_pos_found == True )): #处理右上终点问题
                return {"type":"treasure","pos":(66,97),"config_id":0}
            
            if sum(self.organs_states[1:14]) == 0 and self.last_target is not None and self.last_target["pos"] != (80,50) and self.last_target["type"] != 'end' and ( (self.is_end_pos_found and self.end_pos[0]>=61 and self.end_pos[0]<=82 and self.end_pos[1]>=20 and self.end_pos[1]<=58) or (self.zhongxia == 2 and self.youzhong == 6) ):
                return {"type":"treasure","pos":(80,50),"config_id":0}
            
            if sum(self.organs_states[1:14]) == 0 and self.target_dist[14][0]<60 and self.last_target is not None and self.last_target["pos"] != (57,68) and self.last_target["pos"]!=(61,75) and self.last_target["type"] != 'end' and self.last_target["pos"] != (26,52):
                if sum(self.organs_states[1:14]) == 0 and (not self.end_far):
                    return {"type":"buff","pos":(61,75),"config_id":0}
                else:
                    return {"type":"treasure","pos":(57,68),"config_id":0}  #要在下行代码的上面
            
            #if sum(self.organs_states[1:14]) == 0 and self.last_target is not None and self.last_target["pos"] != (26,52) and self.last_target["type"] != 'end'and self.zuoxia in [5,6] and
            #  not (self.is_end_pos_found and self.end_pos[0]>=30 and self.end_pos[0]<=59 and self.end_pos[1]<=58) :
            if sum(self.organs_states[1:14]) == 0 and self.last_target is not None and self.last_target["pos"] != (26,52) and self.last_target["type"] != 'end'and (self.zuoxia == 5 or ( self.zuoxia == 6 and ( (self.zuoxia_add is not None and self.zuoxia_add in [5,6]) or (self.zuoxia_add is None and not (self.is_end_pos_found and self.end_pos[0]>=30 and self.end_pos[0]<=59 and self.end_pos[1]<=58) )) )):  
                return {"type":"treasure","pos":(26,52),"config_id":0}
            
                
            if sum(self.organs_states[1:14]) == 0:  #表示所有宝箱已经吃完,或者不存在
                return {"type":"end","pos":self.target_dist[14],"config_id":14}
        
            
            
            return {"type":"treasure","pos":target_pos,"config_id":config_id}
    

    def pb2struct(self, frame_state, last_action):


        self.obs = self._obs
        self.extra_info = self._extra_info

        self._obs, self._extra_info = frame_state


        #当前帧信息
        self.step_no = self._obs["frame_state"]["step_no"]
        hero = self._obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        if self.obs is not None:
            self.score_info = self.obs['score_info']
        self._cur_x = hero["pos"]["x"]
        self._cur_z = hero["pos"]["z"]
        if hero["speed_up"]==1: #测试是否1是出于加速状态，0不是
            self._on_buff = True
        else:
            self._on_buff = False
  
        if self.obs==None:
            pass
        else:
            self.cur_x = self.obs["frame_state"]["heroes"][0]["pos"]["x"]
            self.cur_z = self.obs["frame_state"]["heroes"][0]["pos"]["z"]
            self.use_flash = (self.obs['legal_act'][1] - self._obs['legal_act'][1]) == 1
            if self.action is None:
                pass
            else:
                self.hit_wall = self._check_hit_wall()


        #视野域信息
        self.local_view = [self._obs["map_info"][i]["values"] for i in range(len(self._obs["map_info"]))]

        obstacle_map, treasure_map, end_map, buff_map = [], [], [], []
        for sub_list in self.local_view:
            obstacle_map.append([1 if i == 0 else 0 for i in sub_list])
            treasure_map.append([1 if i == 4 else 0 for i in sub_list])
            end_map.append([1 if i == 3 else 0 for i in sub_list])
            buff_map.append([1 if i == 6 else 0 for i in sub_list])

        self.wall_map = np.array(obstacle_map)
        self.end_map = np.array(end_map)
        self.obstacle_flat, self.treasure_flat, self.end_flat, self.buff_flat = [], [], [], []
        for i in obstacle_map:
            self.obstacle_flat.extend(i)
        for i in treasure_map:
            self.treasure_flat.extend(i)
        for i in end_map:
            self.end_flat.extend(i)
        for i in buff_map:
            self.buff_flat.extend(i)
        
        # 局部记忆图
        self.memory_update(self.cur_pos)

        # 闪现是否可用 -- 新增一个字段来代表闪现是否可用，默认初始化的时候可以为True
        if hero['talent']['status'] == 0:
            self.is_flashed = False
        elif hero['talent']['status'] == 1:
            self.is_flashed = True

        # History position
        # 历史位置
        self.history_pos.append(self.cur_pos)   
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

         # End position
        # 终点位置
        for organ in self._obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4:
                end_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
                end_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
                self.end_pos_dir = end_pos_dir
                self.end_pos_dis = end_pos_dis
                if organ["status"] != -1:
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.is_end_pos_found = True
                    # print(f'包含终点的map_info{map_info}')
                # if end_pos is not found, use relative position to predict end_pos
                # 如果终点位置未找到，使用相对位置预测终点位置
                elif (not self.is_end_pos_found) and (
                    self.end_pos is None
                    or self.step_no % 100 == 0
                    or self.end_pos_dir != end_pos_dir
                    or self.end_pos_dis != end_pos_dis
                ):
                    distance = end_pos_dis * 20
                    theta = DirectionAngles[end_pos_dir]
                    delta_x = distance * math.cos(math.radians(theta))
                    delta_z = distance * math.sin(math.radians(theta))

                    self.end_pos = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )

                    self.end_pos_dir = end_pos_dir
                    self.end_pos_dis = end_pos_dis

        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)
        self.end_mhd_dist = abs(self.cur_pos[0]-self.end_pos[0]) + abs(self.cur_pos[1]-self.end_pos[1])
        self.end_far = self.end_mhd_dist>=51

        # History position feature
        # 历史位置特征
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])
        #print(f'当前帧为：{self.step_no},当前起点坐标为{self.cur_pos}，当前预估的终点坐标为：{self.end_pos}，实际的终点坐标为[42,30],当前的方向为：{self.end_pos_dir}')
        self.move_usable = True
        self.last_action = last_action


        #15个物件状态   
        
        self.target_states = []
       
        for organ in self._obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4:
                target = {"type":"end",
                          "config_id":14,
                          "state":self.is_end_pos_found,
                          "pos":self.end_pos}
                # print(f"is_end_pos_found:{self.is_end_pos_found}")
                self.target_states.append(target)
            elif organ["sub_type"] == 1:
                target = {"type":"treasure",
                          "config_id":organ["config_id"],
                          "state":organ["status"],
                          "pos":(organ["pos"]["x"],organ["pos"]["z"])}
                # print(f'宝箱：{target}')
                self.target_states.append(target)
            elif organ["sub_type"] == 2:
                target = {"type":"buff",
                          "config_id":organ["config_id"],
                          "state":organ["status"],
                          "pos":(organ["pos"]["x"],organ["pos"]["z"])}
                #buff是否可收集
                self.buff_flag = target["state"]
                # print(f'buff状态:{self.buff_flag}')
                self.target_states.append(target)

        self.target_dist = [None]*15
        self.organs_states = [0]*15
        for target in self.target_states:
            self.organs_states[target["config_id"]] = target["state"]
            self.target_dist[target["config_id"]] = target["pos"]

        
        #局部视野域卡住，切换目标
        
        if  self.target is not None and (np.maximum(self.local_memory_map[4:7,4:7]-0.9, 0)).sum()>1.0 and self.ka == False: #ka为false表示上次的目标不是buff

            self.target = {"type":"buff","pos":self.target_dist[0],"config_id":15}
            self.ka = True  #ka为true表示上次的目标为buff
        
        # 其他达到buff，也触发ka
        if self.target is not None and self.target["pos"] == (57,68):
            self.ka = True
        
        
        #判断是否到达目标
        arrive = False
        if self.target is not None:
            pos = self.target["pos"]

            if self.target["type"] == 'end':

                if abs(self.cur_pos[0]-pos[0])==0 and abs(self.cur_pos[1]-pos[1])==0:
                    arrive = True
                    self.treasure_map[f"{pos[0]},{pos[1]}"] = 0
             
            else:
                if abs(self.cur_pos[0]-pos[0])<=1 and abs(self.cur_pos[1]-pos[1])<=1:
                    arrive = True
                    self.treasure_map[f"{pos[0]},{pos[1]}"] = 0
                    if pos == (66,97):
                        self.youshang = self.end_pos_dir
                    if pos == (46,55):
                        self.zuoxia = self.end_pos_dir
                    if pos == (64,20):
                        self.zhongxia = self.end_pos_dir
                    if pos == (80,50):
                        self.youzhong = self.end_pos_dir
                    if pos == (38,55):
                        self.zuoxia_add = self.end_pos_dir
                    
               
        
            
   
        #最近欧氏距离切换目标
        if self.target is  None or arrive:  #表示上一次的目标已经达到
    
            if self.target is not None and self.target["pos"] != (57,68) and self._obs["score_info"]["score"] > self.obs["score_info"]["score"]:   #说明到达目标
                self.ka = False

            self.last_target = self.target
            self.target = self.get_target()
     
            
            
        self.target_in_view = False
        #视野域内看到切换目标,优先级最高
        if self.target["type"] == "end":
            pos = self.target_dist[14]
            relative_x = abs(pos[0]-self.cur_pos[0])
            relative_y = abs(pos[1]-self.cur_pos[1])
            if relative_x<=5 and relative_y<=5:
                self.target = {"type":"end","pos":pos,"config_id":14}
                self.target_in_view = True
                             
        else:
            for i in range(1,14):
                pos = self.target_dist[i]
                if self.organs_states[i]!=0:
                    relative_x = abs(pos[0]-self.cur_pos[0])
                    relative_y = abs(pos[1]-self.cur_pos[1])

                    if relative_x<=5 and relative_y<=5:
                        self.target = {"type":"treasure","pos":pos,"config_id":i}
                        self.target_in_view = True
                        break

    

        #下面的特征非规划内容，不需要管

        #目标特征        
        self.target_feature_pos = self._get_pos_feature(True,self.cur_pos,self.target["pos"])
    
        #剩余宝箱数
        self.treasure_num = self._obs["score_info"]["treasure_collected_count"]

        # 目标位置的one-hot编码
        self.pos_row = [0] * 128
        self.pos_row[self.cur_pos[0]] = 1
        self.pos_col = [0] * 128
        self.pos_col[self.cur_pos[1]] = 1
    
         # 判断动态障碍物，没看到是0,独热编码
        for i in range(-5,6):
            if self.dynamic_o!=0:
                break
                
            for j in range(-5,6):
                if i == 0 and j == 0:
                    break
                x = 5+i
                z = 5+j
                if self.wall_map[x][z] == 1:
                    _x = self._cur_x-j
                    _z = self._cur_z+i
                    if _x == 42 and _z>=113 and _z<=117:
                        self.dynamic_o = 1
                    elif _x == 42 and _z>=78 and _z<=81:
                        self.dynamic_o = 2
                    elif _x == 50 and _z>=59 and _z<=63:
                        self.dynamic_o = 3
                    elif _z == 60 and _x>=70 and _x<=75:
                        self.dynamic_o = 4
                    elif _z == 41 and _x>=72 and _x<=77:
                        self.dynamic_o = 5
                    elif _z == 42 and _x>=42 and _x<=46:
                        self.dynamic_o = 6
                    else:
                        self.dynamic_o = 0

        

    def process(self, frame_state, last_action,is_exploit = False):
        self.pb2struct(frame_state, last_action)

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()

        # Feature
        # 特征
        if self.target["type"] == "end":
            feature = np.concatenate([self.cur_pos_norm, self.feature_end_pos,self.feature_end_pos,self.feature_history_pos, legal_action,[self.treasure_num],[self.buff_flag]
            ,[self.target_in_view],self.obstacle_flat,self.end_flat,self.treasure_flat,self.memory_flag,self.pos_row,self.pos_col])
        else:            
            feature = np.concatenate([self.cur_pos_norm, self.target_feature_pos,self.feature_end_pos,self.feature_history_pos, legal_action,[self.treasure_num],[self.buff_flag]
            ,[self.target_in_view],self.obstacle_flat,self.end_flat,self.treasure_flat,self.memory_flag,self.pos_row,self.pos_col])

        if is_exploit:
        
            return (
                feature,
                legal_action
            )
        else:
            return (
                feature,
                legal_action,
                reward_process(self._obs, self._extra_info, self.obs, self.extra_info, self.target, self.target_feature_pos,self.history_pos[0],self.target_in_view,self.hit_wall,self.local_memory_map,self.is_end_pos_found,self.end_pos,self.dynamic_o),
            )

    
    def get_legal_action(self):
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中

        legal_action = [self.move_usable] * self.move_action_num
        
        
        # 添加闪现的合法性
        if self.is_flashed:
            if self.obs==None:
                legal_action[8:] = [False]*8
            else:
                for i in range(8):
                    delta_pos = const.flash_map[i]
                    if self.wall_map[5+delta_pos[0]][5+delta_pos[1]] == 1:
                        if self._cur_x>90:
                            legal_action[8+i] = False

                        elif self._cur_z<=78 and self._cur_z>=70 and self._cur_x<64:
                            legal_action[8+i] = False
                        
                        elif self._cur_z<=113 and self._cur_z>=97 and self._cur_x<64:
                            legal_action[8+i] = False
                        
                        elif self._cur_x>=60 and self._cur_z>=68 and self._cur_z<=97:
                            legal_action[8+i] = False
                        
                        elif self._cur_x<=30 and self._cur_z>=80 and self._cur_z<=116:
                            legal_action[8+i] = False

                        elif self._cur_x>=50 and self._cur_z>=63 and self._cur_z<=75:
                            legal_action[8+i] = False

                        elif self._cur_z>=81 and self._cur_z<=90 and self._cur_x<=62:
                            legal_action[8+i] = False

                        elif self._cur_x>=46 and self._cur_x<=52 and self._cur_z>=20 and self._cur_z<=59:
                            legal_action[8+i] = False
                        
                        elif self._cur_x<=72 and self._cur_x>=60 and self._cur_z>=20 and self._cur_z<=58:
                            legal_action[8+i] = False
                        
                        elif self._cur_x>=48 and self._cur_x<=71 and self._cur_z>=63 and  self._cur_z<=84:
                            legal_action[8+i] = False
                        
                        elif self._cur_x<=34 and self._cur_z<=55:
                            legal_action[8+i] = False

                        elif self._cur_x<=43 and self._cur_z>=55 and self._cur_z<=70:
                            legal_action[8+i] = False

                        elif self._cur_z<=39:
                            legal_action[8+i] = False
                        
                        
                        else:

                            legal_action[8+i] = True
                    else:
                        legal_action[8+i] = False
        else:
            legal_action[8:] = [False] * 8
        
        if self.target['type'] != 'end':
            for i in range(8):
                    delta_pos = const.flash_map[i]
                    if self.end_map[5+delta_pos[0]][5+delta_pos[1]] == 1 or self.end_map[5+delta_pos[0]+delta_pos[0]][5+delta_pos[1]+delta_pos[1]]:
                        legal_action[i] = False
                    else:
                        legal_action[i] = True

        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            legal_action[:8] = [self.move_usable] * 8
            
        return legal_action

    
    def _check_hit_wall(self):  
        if not self._check_has_wall_around(): 
            return False
        dist = abs(self.cur_z-self._cur_z)+abs(self.cur_x-self._cur_x)
        if self.use_flash and dist < 5:
            # print(f"闪现撞墙,dist{dist}")
            return True
        if self._on_buff==True and dist< 1:
            # print(f"buff撞墙,dist{dist}")
            return True
        if self._on_buff==False and dist<1:
            return True
        return False
  
    def _check_has_wall_around(self, alpha=2):
         # 1: wall, 0: no-wall
        x = np.array([5, 5], np.int32)  #当前坐标【5,5】
        for i in range(-1, 2):  #(-1,0,1)
            action = (self.action + i + 8) % 8  #因为i中有负数所以+8是为了%8的意义
            dx = const.delta_pos[action % 8]    #计算相对位移的位置
            for a in range(alpha):  
                tx = x + dx * a
                if self.wall_map[tx[0], tx[1]]: return True
        return False

    # 记忆矩阵的更新函数
    def memory_update(self, cur_pos):
        """
        记忆矩阵更新
        """
        # 全局记忆矩阵
        x,z = cur_pos
        z = 127 - z
        current_value = self.global_memory_map[z, x]
        self.global_memory_map[z, x] = min(1.2, current_value + 0.1)

        # 局部记忆矩阵
        # 计算在全局地图上的源区域边界
        src_top = max(0, z - 5)
        src_bottom = min(128, z + 6)
        src_left = max(0, x - 5)
        src_right = min(128, x + 6)

        # 计算在局部地图上的目标区域边界
        dst_top = src_top - (z - 5)
        dst_bottom = src_bottom - (z - 5)
        dst_left = src_left - (x - 5)
        dst_right = src_right - (x - 5)

        # 从全局地图复制有效区域到局部地图
        self.local_memory_map[dst_top:dst_bottom, dst_left:dst_right] = self.global_memory_map[src_top:src_bottom, src_left:src_right]
        self.memory_flag = self.local_memory_map.flatten()


    
      