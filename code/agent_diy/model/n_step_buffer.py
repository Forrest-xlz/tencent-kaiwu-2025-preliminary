import numpy as np
from collections import deque
from typing import List, Any, Dict, Deque
from agent_diy.feature.definition import SampleData
from kaiwu_agent.utils.common_func import Frame
from agent_diy.conf.conf import Config
import os

class NStepBuffer:
    def __init__(
        self,logger=None
):
        
        self.logger = logger

        # N-step相关
        self.n_step = Config.N_STEP  # N-step的步数
        self.gamma = Config.GAMMA  # 折扣因子
        self.n_step_buffer = deque(maxlen=self.n_step)  # 临时存储最近N个原始frame
        

    def add(self,obs_data,_obs_data,obs_legal_act,_obs_legal_act,act,rew,done,ret):
        
        
        # 将SampleData加入N-step临时缓冲区
        sample = SampleData(obs=obs_data,
        _obs=_obs_data,
        obs_legal=obs_legal_act,
        _obs_legal=_obs_legal_act,
        act=act,
        rew=rew,
        ret = ret,
        done=done,)

        
        self.n_step_buffer.append(sample)
        # 当N-step缓冲区未满N个时，不生成N-step frame
        if len(self.n_step_buffer) < self.n_step:
            raw_frame = Frame(obs=obs_data,
                        _obs=_obs_data,
                        obs_legal=obs_legal_act,
                        _obs_legal=_obs_legal_act,
                        act=act,
                        rew=rew,
                        ret = ret,
                        done = done,)
            
            return raw_frame  # 未满N步，返回原始frame

        # 计算N-step累积奖励、最终状态和终止标志
        n_step_rew, n_step_next_obs, n_step_done = self._get_n_step_info()

        # 生成N-step frame（用最早的obs和act，搭配N-step计算结果）
        earliest_sample = self.n_step_buffer[0]  # 最早的frame
        n_step_frame = Frame(
            obs = earliest_sample.obs,  
            _obs = n_step_next_obs, 
            obs_legal = earliest_sample.obs_legal,  # 使用最早的obs_legal
            _obs_legal = earliest_sample._obs_legal,  # 使用最早的_obs_legal
            act = earliest_sample.act,       # 最早的动作
            rew = n_step_rew,
            ret = n_step_rew,                  
            done = n_step_done                 # N步后的终止标志
        )

        # 返回最早的原始frame（可选，用于兼容需要原始frame的场景）
        return n_step_frame


    def _get_n_step_info(self) -> tuple[float, Any, bool]:
        """从N-step缓冲区的frame中计算累积奖励、最终状态和终止标志"""
        # 从最后一个frame开始初始化
        last_sample = self.n_step_buffer[-1]
        total_rew = last_sample.rew  # 最后一步的奖励
        final_next_obs = last_sample._obs  # 最后一步的下一状态
        final_done = last_sample.done  # 最后一步的终止标志

        # 从倒数第二步开始向前累积（逆序遍历）
        for sample in reversed(list(self.n_step_buffer)[:-1]):
            rew = sample.rew
            current_done = sample.done

            # 累积奖励：当前奖励 + 折扣因子×后续累积奖励（若当前未终止）
            total_rew = rew + self.gamma * total_rew * (1 - current_done)

            # 若当前frame已终止，则更新最终状态和标志（截断后续）
            if current_done:
                final_next_obs = sample._obs
                final_done = current_done

        return total_rew, final_next_obs, final_done


    def size(self) -> int:
        return len(self.n_step_buffer)


    def clear(self):
        
        self.n_step_buffer.clear() 