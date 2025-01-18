import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from PIL import Image
import re
import torch



class CaliDataset(Dataset):
    def __init__(self, channel, depths, distindex, slm_data_path, img_data_path):
        super(CaliDataset, self).__init__()
        self.channel = channel
        self.distindex = distindex
        self.depths = depths
        self.N, self.USLMs, self.images = self.load_data(slm_data_path, img_data_path)
        #USLMs [N, 1, H, W] in RGB images[N, D, C, H, W] in BGR self.dist[D],单位mm

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.USLMs[index], self.images[index]
    
    def load_data(self, slm_data_path, img_data_path):
        print("Loading data from " + slm_data_path + " and")
        print("              and " + img_data_path + "......")
        USLMs_dict = {}
        for filename in os.listdir(slm_data_path):
            file_path = os.path.join(slm_data_path, filename)
            index = int(filename[0])
            with Image.open(file_path) as img:
                USLMs_dict[index] = img.copy() 
        USLMs = [None] * len(USLMs_dict)
        for key, value in USLMs_dict.items():
            USLMs[key] = cv2.cvtColor(np.array(value), cv2.COLOR_BGR2RGB)
        USLMs = np.array(USLMs).transpose(0, 3, 1, 2)#[N, H, W, 1] - [N, 1, H, W]
        USLMs = USLMs[:, self.channel, :, :]
        Images_dict = {}
        pattern = re.compile(r'(\d+)_d(\d+(\.\d+)?).png')
        for filename in os.listdir(img_data_path):
            file_path = os.path.join(img_data_path, filename)
            match = pattern.match(filename)
            if match:
                index_scene = int(match.group(1))  
                dist = float(match.group(2))
                index_dist = self.distindex[dist]
            with Image.open(file_path) as img:
                if(index_scene in Images_dict):
                    Images_dict[index_scene][index_dist] = np.array(img.copy())
                    Images_dict[index_scene][index_dist] = cv2.cvtColor(Images_dict[index_scene][index_dist], cv2.COLOR_BGR2RGB)

                else:
                    Images_dict[index_scene] = {}
                    Images_dict[index_scene][index_dist] = np.array(img.copy())
                    Images_dict[index_scene][index_dist] = cv2.cvtColor(Images_dict[index_scene][index_dist], cv2.COLOR_BGR2RGB)
        
        Images = [[None] * self.depths] * len(Images_dict)
        for outer_key, inner_key in Images_dict.items():
            for inner_key, value in Images_dict[outer_key].items():
                Images[outer_key][inner_key] = value #转化为RGB模式
        Images = np.array(Images).transpose(0, 1, 4, 2, 3) # [N, D, H, W, 1] - [N, D, 1, H, W]
        
        Images = Images[:, :, self.channel, :, :] #self.opts.calibrate_channel: 0,1,2分别代表红绿蓝
        #Images = np.expand_dims(Images, axis = 2)
        self.grid_size = Images.shape
        print("Grid size:", self.grid_size)
        print("Loading channel : ", self.channel)
        print("Loading finished!")
        return len(Images_dict), torch.from_numpy(USLMs), torch.from_numpy(Images)