import cv2
import numpy as np
from Loss import calculate_psnr
from Loss import SSIM
import torch
import torch.nn.functional as F

pic_red_path = "/data/wyw/lwj/XPR_holo/runs/scene_2/red_ASM_NoXPR_h1080_r0.0300/Figures_out/1250_e4_d.png"
pic_green_path = "/data/wyw/lwj/XPR_holo/runs/scene_2/green_ASM_NoXPR_h1080_r0.0300/Figures_out/1250_e4_d.png"
pic_blue_path = "/data/wyw/lwj/XPR_holo/runs/scene_2/blue_ASM_NoXPR_h1080_r0.0300/Figures_out/1250_e4_d.png"
scene = "scene2_4_d_NOXPR_2k"
target_path = "/data/wyw/lwj/XPR_holo/data/scene_2_4k/4_gt.png"# 2K 也可以直接和 4K 照片做对比


def merge_pic(pic_red_path, pic_green_path, pic_blue_path, path):
    
    pic_red = cv2.imread(pic_red_path)[:,:,0]
    pic_green = cv2.imread(pic_green_path)[:,:,0]
    pic_blue = cv2.imread(pic_blue_path)[:,:,0]

    bgr_channels = np.array([pic_red, pic_green, pic_blue]).transpose(1, 2, 0)    
    cv2.imwrite(path, bgr_channels)
    return bgr_channels


if __name__ == "__main__":
    ssim = SSIM()
    image = merge_pic(pic_red_path, pic_green_path, pic_blue_path, "/data/wyw/lwj/XPR_holo/runs/scene_2/" + scene + ".png")
    target_image = cv2.imread(target_path)
    if(image.shape[1] < target_image.shape[1]):
        image = np.repeat(image, 2, axis=0)
        image = np.repeat(image, 2, axis=1)
    snr_value = calculate_psnr(target_image,image)
    ssimloss = ssim(torch.tensor(np.transpose(image, (2, 1, 0))).type(torch.float32), torch.tensor(np.transpose(target_image, (2, 1, 0))).type(torch.float32)).mean()
    print("snr of merged image:{0:.5f}".format(snr_value))
    print("ssim of merged image:{0:.5f}".format(float(ssimloss)))