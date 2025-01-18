from prop_ideal import SerialProp
import torch
import torch.nn as nn
import prop_ideal
import numpy as np
import torch.nn.functional as F
import torch.fft as tfft
import cv2
import os

class XPR_Model(nn.Module):
    def __init__(self, res_target, Model, depth_num):
        super(XPR_Model, self).__init__()
        shape_CGH = res_target // 2
        self.CGH_l_u = nn.Parameter(torch.randn((1, shape_CGH[0], shape_CGH[1]))).to(dtype = torch.float32)
        self.CGH_u = nn.Parameter(torch.randn((1, shape_CGH[0], shape_CGH[1]))).to(dtype = torch.float32)
        self.CGH = nn.Parameter(torch.randn((1, shape_CGH[0], shape_CGH[1]))).to(dtype = torch.float32)
        self.CGH_l = nn.Parameter(torch.randn((1, shape_CGH[0], shape_CGH[1]))).to(dtype = torch.float32)
        self.scale_factor = nn.Parameter(torch.ones(depth_num, 1, 1)).to(dtype = torch.float32)
        nn.init.xavier_uniform_(self.CGH_l_u)
        nn.init.xavier_uniform_(self.CGH_u)
        nn.init.xavier_uniform_(self.CGH)
        nn.init.xavier_uniform_(self.CGH_l)
        self.Model = Model
        self.res_target = tuple(res_target)

    def forward(self):
        Amplitude1 = F.interpolate(torch.abs(self.Model(self.CGH_l_u)).unsqueeze(1), mode = "nearest", size = self.res_target)
        Amplitude2 = F.interpolate(torch.abs(self.Model(self.CGH_u)).unsqueeze(1), mode = "nearest", size = self.res_target)
        Amplitude3 = F.interpolate(torch.abs(self.Model(self.CGH)).unsqueeze(1), mode = "nearest", size = self.res_target)
        Amplitude4 = F.interpolate(torch.abs(self.Model(self.CGH_l)).unsqueeze(1), mode = "nearest", size = self.res_target)
        
        A1_padded = F.pad(Amplitude1, (0, 1, 0, 1), mode = 'constant')
        A2_padded = F.pad(Amplitude2, (0, 0, 0, 1), mode = 'constant')
        A4_padded = F.pad(Amplitude4, (0, 1, 0, 0), mode = 'constant')


        A = self.scale_factor * torch.sqrt((A1_padded[:, :, 1:,1: ]**2 + A2_padded[:, :, 1:, :]**2+ 
                                                   Amplitude3**2 + A4_padded[:, :, :, 1: ]**2)/4)
        return A


class Origin_Model(nn.Module):
    def __init__(self, res_target, Model, depth_num):
        super(Origin_Model, self).__init__()
        self.CGH = nn.Parameter(torch.randn((1, *res_target))).to(dtype = torch.float32)
        self.scale_factor = nn.Parameter(torch.ones(depth_num, 1, 1)).to(dtype = torch.float32)
        self.Model = Model
        self.res_target = tuple(res_target)    

    def forward(self):
    
        A = torch.abs(self.Model(self.CGH)).unsqueeze(1)

        A = self.scale_factor * A
        return A
