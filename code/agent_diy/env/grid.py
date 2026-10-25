import numpy as np
import json
import cv2
from agent_diy.conf.conf import Config
import os
from copy import deepcopy
import time

class Grid:

    def __init__(self,logger = None,monitor = None):
        self.GRID = None
        self.grid_file_path = Config.GRID_FILE_PATH
        self.logger = logger
        self.monitor = monitor
        self.grid = None
        self.grid_save_path = Config.GRID_SAVE_PATH_TRAIN
        self.treasure_pos = None #字典存储每个宝箱的id，x，z
      

    def get_grid(self,obs):
        self.raw_obs = obs
        try:
            with open(self.grid_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)  # 加载JSON数据为字典
                self.logger.info(f"Grid file load successfully!")
        except FileNotFoundError:
            self.logger.info(f"Error! grid file not find: {self.grid_file_path}")
            exit()
        except json.JSONDecodeError:
            self.logger.info(f"Error! file: {self.grid_file_path} is not json file")
            exit()
        except Exception as e:
            self.logger.info(f"Fail to load:{str(e)}")
            

        # 2. 提取Flags列表和尺寸信息
        flags = data.get("Flags", [])  # 获取Flags列表，默认空列表
        width = data.get("Width", 128)   # 从JSON中获取宽度（应为64）
        height = data.get("Height", 128) # 从JSON中获取高度（应为64）

        # 3. 验证Flags长度是否符合64x64（64*64=4096）
        if len(flags) != width * height:
            self.logger.info(f"Error! grid len should be {width*height},but {len(flags)}")
            exit()

        self.grid = np.array(flags).reshape(height,width)
        self.GRID = deepcopy(self.grid)


        #获取每个宝箱的pos
        treasure_pos = []  # 存储所有宝箱的相对距离（x方向, z方向）
        frame_state = self.raw_obs["frame_state"]
        for organ in frame_state["organs"]:
            if organ["sub_type"] == 4:
                self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                self.grid[organ["pos"]["z"]][organ["pos"]["x"]] = 6  # 终点位置标记为6
            if organ["sub_type"] == 3:
                self.start_pos = (organ["pos"]["x"], organ["pos"]["z"])
                self.grid[organ["pos"]["z"]][organ["pos"]["x"]] = 5  # 起点位置标记为5
            if organ["sub_type"] == 2:
                self.buff_pos = (organ["pos"]["x"], organ["pos"]["z"])
                self.grid[organ["pos"]["z"]][organ["pos"]["x"]] = 4  # buff位置标记为4
            if organ["sub_type"] == 1:
                treasure_pos.append((organ["pos"]["x"], organ["pos"]["z"]))
                self.grid[organ["pos"]["z"]][organ["pos"]["x"]] = 3  # 宝箱位置标记为3

        self.treasure_pos = treasure_pos
            

    def update_grid(self,x,z,values):
        if values==2:
            if self.grid[z][x] == 3 or self.grid[z][x] == 4 or self.grid[z][x] == 5 or self.grid[z][x] == 6:   
                return 
            
            else:
                self.grid[z][x] = 2
        else:
            self.grid[z][x] = values
 

    def reset_grid(self,logger):
        self.grid = deepcopy(self.GRID)
        

    def save_img(self, episode,x,z):
        
        # 基础设置：中等缩放（20倍，兼顾清晰和文件大小）
        scale = 20
        
        # 默认颜色映射（黑白）：0=黑（障碍），1=白（可行区）
        # 优化后颜色映射：高饱和度+明确色系区分，关键元素更醒目
        default_color = {
            0: (50, 50, 50),        # 障碍
            1: (240, 240, 240),     # 可行区
            2: (255,0,0),       # 智能体路径
            3: (0,0,0),       # 未收集宝箱
            4: (128,0,128),       # 未收集buff
            5: (0,255,0),       # 起点
            6: (0,0,255),       # 终点
            7: (0,255,0),      # 已收集宝箱
            8: (34, 139, 34)        # 已收集buff
        }
        color_map = default_color
        
        # 创建三通道图像（BGR格式，OpenCV默认）
        img = np.zeros((self.grid.shape[0], self.grid.shape[1], 3), dtype=np.uint8)
        
        # 填充颜色（按值匹配）
        for value, color in color_map.items():
            mask = self.grid == value
            img[mask] = color  # 直接赋值BGR颜色
        
        # 缩放图像（最近邻插值，保证颜色边界清晰）
        img = cv2.resize(
            img,
            (self.grid.shape[1] * scale, self.grid.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST
        )
        
        if not os.path.exists(self.grid_save_path):
            self.logger.info(f"save_path不存在")
            self.logger.info(os.path.abspath(self.grid_save_path))
            return 
        else:
            img_file_name = f"{self.grid_save_path}/episode-{str(episode)}-{str(x)}-{str(z)}-{str(time.time())}.png"
            # 保存图像
            try:
                # 保存为PNG，压缩等级0（无损）
                cv2.imwrite(img_file_name, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])   #png_compression: PNG压缩级别（0-9，值越高文件越小，默认6）
                self.logger.info(f"地图已保存至 {self.grid_save_path}")
                return True
            except Exception as e:
                self.logger.info(f"保存图像失败：{str(e)}")
                return False
