import configargparse
import utils.utils as utils
from prop_ideal import Propagation
from prop_ideal import SerialProp
from XPR_Model import XPR_Model, Origin_Model
from dataloader import XPRDataset
import os
from torch import optim
from trainer_kit import *
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import numpy as np
from Loss import *

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', is_config_file=True, help='Path to config file.', default = "XPR_holo/config.yaml")
p.add_argument("--device",type=str, default="cpu",
                        help='cuda id used in this network')  
#训练模式设置，选择是否重新开始或者清除实验
p.add_argument('--restart', type = bool, help='delete old weight and retrain', default= False)
p.add_argument('--clean', type = bool, help='delete old weight without start training process',default=False)
p.add_argument('--seed', type=int, default=123, help='random seed')
p.add_argument('--run_path', type=str, default="",
                            help='exp directory which store in runs/')

#数据集相关设置，场景，图片大小，训练的channel。
p.add_argument('--scene', type=str, default="", help='name of the picture scene')
p.add_argument('--grid_size', type = int ,nargs = 3, default=[1, 2160, 3840])
p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2')
p.add_argument('--scene_path', type = str, default = "", help = 'image source of the scene')

#模型细节设置，包括选用何种传播模型，相关模型存储位置
p.add_argument('--model_name', type = str, default = "ASM", help = 'The propagation model you choose')
p.add_argument('--use_xpr', type = bool, default = False)
p.add_argument('--calibration_path', type=str, default=f'./calibration',
               help='If use calibrate model, refer to Directory where calibration phases are being stored.')

#训练过程设置，选择学习率，选择周期数，选择是否动态调整学习率等
p.add_argument('--lr', type=float, default=5e-3, help='Learning rate for phase')
p.add_argument('--step_lr', type=utils.str2bool, default=True, help='Use of lr scheduler')
p.add_argument('--epochs', type=int, default=5000, help='total epochs to train')

opt = p.parse_args()
channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]





#硬件相关设置及传播距离设置
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]
feature_size = (7.2 * um, 7.2 * um)
F_aperture = 0.5
slm_res = (1080, 1920)
prop_dist = [0, 1, 2, 3, 4, 5, 6, 7]
prop_dists_from_wrp = [4, 4, 4, 4, 4, 4 ,4, 4] 
prop_dist = [x * mm for x in prop_dist]
prop_dists_from_wrp = [x * mm for x in prop_dists_from_wrp]


#初始化XPR传播模型
if (opt.use_xpr):
    print("Initializing the " + opt.model_name + " model")
    if(opt.model_name == 'ASM'):
        forward = SerialProp(prop_dists_from_wrp, wavelength, feature_size, prop_type= 'ASM', 
                         F_aperture = F_aperture, prop_dists_from_wrp= [a - b for a, b in zip(prop_dist, prop_dists_from_wrp)])
    print("Finished")
    print("Initializing the XPR Model")
    XPR_model = XPR_Model(np.array(opt.grid_size)[-2: ], forward, opt.grid_size[0])
    print("Finished!")
else:
    print("Initializing the " + opt.model_name + " model")
    if(opt.model_name == 'ASM'):
        feature_size = tuple(np.array(feature_size)/ 2)
        forward = SerialProp(prop_dists_from_wrp, wavelength, feature_size, prop_type= 'ASM', 
                         F_aperture = F_aperture, prop_dists_from_wrp= [a - b for a, b in zip(prop_dist, prop_dists_from_wrp)])
    print("Finished")
    XPR_model = Origin_Model(np.array(opt.grid_size)[-2: ], forward, opt.grid_size[0])

#初始化数据集
print("Loading data from " + opt.scene_path)
print("Loading channel : ", chan_str)
XPR_set = XPRDataset(opt.scene_path, opt.channel)
print("Loading finished!")

if(opt.use_xpr):
    config_name = chan_str + '_' + opt.model_name + '_' + 'XPR' + '_'+ 'h{0:d}_r{1:.4f}'.format(
            opt.grid_size[-2],
            opt.lr
            )
else:
    config_name = chan_str + '_' + opt.model_name + '_' + 'NoXPR' + '_'+ 'h{0:d}_r{1:.4f}'.format(
            opt.grid_size[-2],
            opt.lr
            )    
log_path =  opt.run_path + "/" + opt.scene + "/" + config_name + "/"

#重新开始或者删除日志记录
if opt.restart or opt.clean:
    os.system("rm -rf " + log_path)
if opt.clean:
    exit()

os.makedirs(log_path, exist_ok=True)
start_epoch = 0

#设置损失函数
mseLoss = nn.MSELoss()
ssim = SSIM()

#设置优化器
optimizer = optim.Adam(XPR_model.parameters(), lr= opt.lr)

#导入checkpoint
if os.path.exists(log_path + "/ckpt.pt"):
    print("exist checkpoint")
    start_epoch = loadFromCheckpoint(log_path + "/ckpt.pt", XPR_model, optimizer)

total = sum([param.nelement() for param in XPR_model.parameters()])

print("Number of parameter: %.2fM" % (total / 1e6))

if __name__ == "__main__":
    print("Start Training!")
    ckpt = log_path + "/ckpt.pt"
    optimizer.zero_grad() 

    #将参数,优化器,target,Loss函数放入GPU
    XPR_model = XPR_model.to(opt.device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(opt.device)
    A_target = torch.tensor(XPR_set.images).unsqueeze(1).to(opt.device) / 255
    mseLoss = mseLoss.to(opt.device)
    ssimLoss = ssim.to(opt.device)
    backupConfigAndCode(opt.clean)
    writer = SummaryWriter(log_path)

    for epoch in range(start_epoch, opt.epochs):
        optimizer.zero_grad()    
        if (epoch+1) % 5000 == 0:
            lr_decay(optimizer)
        A = XPR_model.forward()
        loss_epoch = mseLoss(A, A_target) + 0.1 * ssim(A, A_target).mean()
        if((epoch+1) % 50 == 0):
            print("{0:d}epoch done, with".format(epoch + 1))
            print("\ttotal loss :{0:.5f}".format(loss_epoch))

        loss_epoch.backward()
        optimizer.step()
        if ((epoch + 1) % 250 == 0):
            checkpoint(ckpt, XPR_model, optimizer, epoch)
            vis_rgbd_sive(log_path, A, epoch)
        writer.add_scalar('loss', loss_epoch, epoch + 1)

    writer.close()
    print("Training done")
    print("Saving CGH and Model")
    os.makedirs(log_path + 'Model_out/', exist_ok=True)
    torch.save(XPR_model, log_path + 'Model_out/' + 'XPR_model.pt')
    
    if(opt.use_xpr):
        CGH_l_u = Tensor2CGH(XPR_model.CGH_l_u)
        CGH_u = Tensor2CGH(XPR_model.CGH_u)
        CGH = Tensor2CGH(XPR_model.CGH)
        CGH_l = Tensor2CGH(XPR_model.CGH_l)
        cv2.imwrite(log_path + 'Model_out/' + 'CGH_l_u.png', CGH_l_u)
        cv2.imwrite(log_path + 'Model_out/' + 'CGH_u.png', CGH_u)
        cv2.imwrite(log_path + 'Model_out/' + 'CGH.png', CGH)
        cv2.imwrite(log_path + 'Model_out/' + 'CGH_l.png', CGH_l)
    else:
        CGH = Tensor2CGH(XPR_model.CGH)
        cv2.imwrite(log_path + 'Model_out/' + 'CGH.png', CGH)
    loss_SSIM_final = ssimLoss(A, A_target).mean()
    psnr = calculate_psnr(np.array(A.detach().cpu()), np.array(A_target.detach().cpu()))
    mseloss = mseLoss(A, A_target)
    
    if(opt.use_xpr):
        A_l_u = XPR_model.scale_factor*torch.abs(XPR_model.Model(XPR_model.CGH_l_u)).unsqueeze(1)
        A_l = XPR_model.scale_factor*torch.abs(XPR_model.Model(XPR_model.CGH_l)).unsqueeze(1)
        A = XPR_model.scale_factor*torch.abs(XPR_model.Model(XPR_model.CGH)).unsqueeze(1)
        A_u = XPR_model.scale_factor*torch.abs(XPR_model.Model(XPR_model.CGH_u)).unsqueeze(1)
        print("saving raw cgh out amplitude")
        vis_rgbd_test(log_path + 'Figures_out/', A_l_u, "l_u")
        vis_rgbd_test(log_path + 'Figures_out/', A_l, "l")
        vis_rgbd_test(log_path + 'Figures_out/', A, "")
        vis_rgbd_test(log_path + 'Figures_out/', A_u, "u")
    print('saving finished')
    print("final ssim loss:{0:.4f}".format(loss_SSIM_final))
    print("final_mse_loss:{0:.4f}".format(mseloss))
    print("final_psnr:{0:.4f}".format(psnr))


    

