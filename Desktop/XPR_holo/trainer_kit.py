import torch
import os 
import cv2
import numpy as np

def loadFromCheckpoint(file, model, optimizer):
        checkpoint = torch.load(file, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        return start_epoch 

def checkpoint(file, model, optimizer, epoch):
        print("Checkpointing Model @ Epoch %d ..." % epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, file)

def lr_decay(optimizer):
    print('decay learning rate')
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5

def backupConfigAndCode(clean):
    if clean:
        return 

def lr_decay(optimizer):
    print('decay learning rate')
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5

def vis_rgbd_sive(write_path, image_batch, epoch = 0):
    os.makedirs(write_path + 'Figures_out/', exist_ok=True)
    for i in range(image_batch.shape[0]):
        if(image_batch.shape[1] == 1):
            image = np.array(image_batch[i].detach().cpu() * 255).astype(np.uint8).transpose(1, 2, 0)
            image_path = write_path + 'Figures_out/' + '{0:d}_e{1:d}_d.png'.format(epoch + 1, i)
            cv2.imwrite(image_path, image)

def vis_rgbd_test(path, image_batch, name):
    print(image_batch.shape)
    for i in range(image_batch.shape[0]):
        image = np.array(image_batch[i].detach().cpu() * 255).astype(np.uint8).transpose(1, 2, 0)
        image_path = path + '{0:d}_d_'.format(i) + name + ".png"
        cv2.imwrite(image_path, image)    

def Tensor2CGH(CGH_tensor):
    cgh_norm_png = ((np.array(CGH_tensor.detach().cpu()) + np.pi) % (2 * np.pi) / (2 * np.pi) * 255).astype(int).transpose(1, 2, 0)
    return cgh_norm_png