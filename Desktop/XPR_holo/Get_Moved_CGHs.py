import cv2
import numpy as np


def process_image(input_image_path, output_image_path, opt):
    # 读取图像
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    # 检查图像是否读取成功
    if image is None:
        raise ValueError(f"Failed to read image at {input_image_path}")
    # 获取原图像的尺寸
    height, width = image.shape[:2]
    # 新图像的尺寸为原图像尺寸的两倍
    new_height, new_width = height * 2, width * 2
    # 进行插值操作
    Amplitude1 = cv2.resize(np.abs(image), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    Amplitude1 = Amplitude1[..., np.newaxis]
    # 进行填充操作
    if(opt == "l_u"):
        A1_padded = cv2.copyMakeBorder(Amplitude1, 0, 1, 0, 1, cv2.BORDER_CONSTANT)
        A1 = A1_padded[1:,1: ]
    elif(opt == "u"):
        A1_padded = cv2.copyMakeBorder(Amplitude1, 0, 0, 0, 1, cv2.BORDER_CONSTANT)
        A1 = A1_padded[:, 1: ]
    elif(opt == "l"):
        A1_padded = cv2.copyMakeBorder(Amplitude1, 0, 1, 0, 0, cv2.BORDER_CONSTANT)
        A1 = A1_padded[1, : ]                
    # 保存图像
    cv2.imwrite(output_image_path, A1)


# 调用函数，传入输入图像路径和输出图像路径
process_image("/data/wyw/lwj/XPR_holo/runs/scene_2/green_ASM_XPR_h2160_r0.0300/Model_out/CGH_l_u.png", 
              "/data/wyw/lwj/XPR_holo/runs/scene_2/green_ASM_XPR_h2160_r0.0300/Model_out/CGH_l_u_m.png",
              "l_u")
process_image("/data/wyw/lwj/XPR_holo/runs/scene_2/green_ASM_XPR_h2160_r0.0300/Model_out/CGH_u.png", 
              "/data/wyw/lwj/XPR_holo/runs/scene_2/green_ASM_XPR_h2160_r0.0300/Model_out/CGH_u_m.png",
              "u")
process_image("/data/wyw/lwj/XPR_holo/runs/scene_2/green_ASM_XPR_h2160_r0.0300/Model_out/CGH_l.png", 
              "/data/wyw/lwj/XPR_holo/runs/scene_2/green_ASM_XPR_h2160_r0.0300/Model_out/CGH_l_m.png",
              "u")