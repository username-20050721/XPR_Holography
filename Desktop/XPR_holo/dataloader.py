import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from PIL import Image

class XPRDataset(Dataset):
    def __init__(self, data_folder, channel):
        super(XPRDataset, self).__init__()
        self.channel = channel
        self.N, self.images = self.load_data(data_folder)
        #USLMs [N, 1, H, W] in RGB images[N, D, C, H, W] in BGR self.dist[D],单位mm

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.USLMs[index], self.images[index]
    
    def load_data(self, data_folder):
        image_dict = {}
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            index = int(filename[0])
            with Image.open(file_path) as img:
                image_dict[index] = img.copy() 
        images = [None] * len(image_dict)
        for key, value in image_dict.items():
            images[key] = cv2.cvtColor(np.array(value), cv2.COLOR_BGR2RGB)
        images = np.array(images).transpose(0, 3, 1, 2)#[D, H, W, C] -> [D, C, H, W]
        images  = images[:, self.channel, :, :]
        
        shape = images.shape[-2: ]
        print("target resolution: ", shape)
        return len(images), images