import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import Resize, Compose, Normalize
import torch.nn.functional as F 
import numpy as np
import cv2

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(8, 8)
        self.mu_y_pool   = nn.AvgPool2d(8, 8)
        self.sig_x_pool  = nn.AvgPool2d(8, 8)
        self.sig_y_pool  = nn.AvgPool2d(8, 8)
        self.sig_xy_pool = nn.AvgPool2d(8, 8)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
    

class Rfft2d(nn.Module):
    """
    Blockwhise 2D FFT
    for fixed blocksize of 8x8
    """
    def __init__(self, blocksize=8, interleaving=False):
        """
        Parameters:
        """
        super().__init__() # call super constructor
        
        self.blocksize = blocksize
        self.interleaving = interleaving
        if interleaving:
            self.stride = self.blocksize // 2
        else:
            self.stride = self.blocksize
        
        self.unfold = torch.nn.Unfold(kernel_size=self.blocksize, padding=0, stride=self.stride)
        return
        
    def forward(self, x):
        """
        performs 2D blockwhise DCT
        
        Parameters:
        x: tensor of dimension (N, 1, h, w)
        
        Return:
        tensor of dimension (N, k, b, b/2, 2)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block real FFT coefficients. 
        The last dimension is pytorches representation of complex values
        """
        
        (N, C, H, W) = x.shape
        assert (C == 1), "FFT is only implemented for a single channel"
        assert (H >= self.blocksize), "Input too small for blocksize"
        assert (W >= self.blocksize), "Input too small for blocksize"
        assert (H % self.stride == 0) and (W % self.stride == 0), "FFT is only for dimensions divisible by the blocksize"
        
        # unfold to blocks
        x = self.unfold(x)
        # now shape (N, 64, k)
        (N, _, k) = x.shape
        
        x = x.view(-1,self.blocksize,self.blocksize,k).permute(0,3,1,2)
        # # now shape (N, #k, b, b)
        # # perform DCT
        # coeff = torch.fft.rfft(x, dim=2)
        ans = torch.fft.rfft2(x)
        
        return torch.stack((ans.real, ans.imag), -1)
        # return coeff / self.blocksize**2
    
    def inverse(self, coeff, output_shape):
        """
        performs 2D blockwhise inverse rFFT
        
        Parameters:
        output_shape: Tuple, dimensions of the outpus sample
        """
        if self.interleaving:
            raise Exception('Inverse block FFT is not implemented for interleaving blocks!')
        
        # perform iRFFT
        x = torch.irfft(coeff, signal_ndim=2, signal_sizes=(self.blocksize,self.blocksize))
        (N, k, _, _) = x.shape
        x = x.permute(0,2,3,1).view(-1, self.blocksize**2, k)
        x = F.fold(x, output_size=(output_shape[-2], output_shape[-1]), kernel_size=self.blocksize, padding=0, stride=self.blocksize)
        return x * (self.blocksize**2)


class PerceptualLossVGG19(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG19, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19.eval()
        for parm in self.vgg19.parameters():
            parm.requires_grad = False        
        self.feature_module1 = self.vgg19.features[0 : 4]
        self.feature_module2 = self.vgg19.features[0 : 9]  
        self.feature_module1 = self.feature_module1
        self.feature_module2 = self.feature_module2
        self.preprocess = Compose([
            Resize((224, 224)),  # 调整图像尺寸到 224x224
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
            ])
    
    def forward(self, x, y):
        if x.shape[0] == 1 and x.dim() == 3: #(C,H,W)
            x = torch.cat((x, x, x), 0)
            y = torch.cat((y, y, y), 0)
        if x.shape[1] == 1 and x.dim() == 4: #(1,C,H,W)
            x = torch.cat((x, x, x), 1)
            y = torch.cat((y, y, y), 1)            
        x = self.preprocess(x)
        y = self.preprocess(y)
        if(x.dim() == 3):
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        loss1 = torch.mean(torch.abs(self.feature_module1(x) - self.feature_module1(y)))
        loss2 = torch.mean(torch.abs(self.feature_module2(x) - self.feature_module2(y)))
        return loss1 * 0.5 + loss2 * 0.5

def softmax(a, b, factor=1):
    concat = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
    softmax_factors = F.softmax(concat * factor, dim=-1)
    return a * softmax_factors[:,:,:,:,0] + b * softmax_factors[:,:,:,:,1]



class WatsonDistanceFft(nn.Module):
    
    """
    Loss function based on Watsons perceptual distance.
    Based on FFT quantization
    """
    def __init__(self, blocksize=8, trainable=False, reduction='sum', size_average=True):
        """
        Parameters:
        blocksize: int, size of the Blocks for discrete cosine transform 
        trainable: bool, if True parameters of the loss are trained and dropout is enabled.
        reduction: 'sum' or 'none', determines return format
        """
        self.EPS = 1e-10
        super().__init__()
        self.trainable = trainable
        
        # input mapping
        blocksize = torch.as_tensor(blocksize)
        
        # module to perform 2D blockwise rFFT
        self.add_module('fft', Rfft2d(blocksize=blocksize.item(), interleaving=False))
    
        # parameters
        self.weight_size = (blocksize, blocksize // 2 + 1)
        self.blocksize = nn.Parameter(blocksize, requires_grad=False)
        # init with uniform QM
        self.t_tild = nn.Parameter(torch.zeros(self.weight_size), requires_grad=trainable)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=trainable) # luminance masking
        w = torch.tensor(0.2) # contrast masking
        self.w_tild = nn.Parameter(torch.log(w / (1- w)), requires_grad=trainable) # inverse of sigmoid
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=trainable) # pooling
        
        # phase weights
        self.w_phase_tild = nn.Parameter(torch.zeros(self.weight_size) -2., requires_grad=trainable)
        
        # dropout for training
        self.dropout = nn.Dropout(0.5 if trainable else 0)
        
        # reduction
        self.reduction = reduction
        if reduction not in ['sum', 'none']:
            raise Exception('Reduction "{}" not supported. Valid values are: "sum", "none".'.format(reduction))
        self.size_average = size_average

    @property
    def t(self):
        # returns QM
        qm = torch.exp(self.t_tild)
        return qm
    
    @property
    def w(self):
        # return luminance masking parameter
        return torch.sigmoid(self.w_tild)
    
    @property
    def w_phase(self):
        # return weights for phase
        w_phase =  torch.exp(self.w_phase_tild)
        # set weights of non-phases to 0
        if not self.trainable:
            w_phase[0,0] = 0.
            w_phase[0,self.weight_size[1] - 1] = 0.
            w_phase[self.weight_size[1] - 1,self.weight_size[1] - 1] = 0.
            w_phase[self.weight_size[1] - 1, 0] = 0.
        return w_phase
    
    def forward(self, input, target):
        # fft
        N, K, H, W = input.shape
        c0 = self.fft(target)
        c1 = self.fft(input)
        N, K, H, W, _ = c0.shape
        
        # get amplitudes
        c0_amp = torch.norm(c0 + self.EPS, p='fro', dim=4)
        c1_amp = torch.norm(c1 + self.EPS, p='fro', dim=4)
        
        # luminance masking
        avg_lum = torch.mean(c0_amp[:,:,0,0])
        t_l = self.t.view(1, 1, H, W).expand(N, K, H, W)
        t_l = t_l * (((c0_amp[:,:,0,0] + self.EPS) / (avg_lum + self.EPS)) ** self.alpha).view(N, K, 1, 1)
        
        # contrast masking
        s = softmax(t_l, (c0_amp.abs() + self.EPS)**self.w * t_l**(1 - self.w))
        
        # pooling
        watson_dist = (((c0_amp - c1_amp) / s).abs() + self.EPS) ** self.beta
        watson_dist = self.dropout(watson_dist) + self.EPS
        watson_dist = torch.sum(watson_dist, dim=(1,2,3))
        watson_dist = watson_dist ** (1 / self.beta)
        
        # get phases
        c0_phase = torch.atan2( c0[:,:,:,:,1], c0[:,:,:,:,0] + self.EPS) 
        c1_phase = torch.atan2( c1[:,:,:,:,1], c1[:,:,:,:,0] + self.EPS)
        
        # angular distance
        phase_dist = torch.acos(torch.cos(c0_phase - c1_phase)*(1 - self.EPS*10**3)) * self.w_phase # we multiply with a factor ->1 to prevent taking the gradient of acos(-1) or acos(1). The gradient in this case would be -/+ inf
        phase_dist = self.dropout(phase_dist)
        phase_dist = torch.sum(phase_dist, dim=(1,2,3))
        
        # perceptual distance
        distance = watson_dist + phase_dist
        
        # reduce
        if self.reduction == 'sum':
            distance = torch.sum(distance)
        if self.size_average:
            distance = distance / (N * K * H * W)
        
        return distance


def calculate_psnr(img_origin, img_now):

    # 计算两张图像之间的均方误差 (MSE)
    mse = ((img_origin - img_now) ** 2).mean()

    # 计算峰值信噪比 (PSNR)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr