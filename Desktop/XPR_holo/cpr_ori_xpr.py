import cv2
from Loss import * 
if __name__ == "__main__":
    channel = 1
    image = cv2.imread("/data/wyw/lwj/XPR_holo/runs/scene_2/green_ASM_NoXPR_h1080_r0.0300/Figures_out/1000_e4_d.png")[:,:,0][:,:,np.newaxis] 
    target = cv2.imread("/data/wyw/lwj/XPR_holo/data/scene_2_4k/4_gt.png")[:,:, channel][:,:,np.newaxis]
    if(np.array(image.shape)[0] == (np.array(target.shape)[0] //2)):
        image= cv2.resize(image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis]
    psnr = calculate_psnr(target, image)
    ssim = SSIM()
    ssimloss = ssim(torch.tensor(image).squeeze().unsqueeze(0)/255, torch.tensor(target).squeeze().unsqueeze(0)/255).mean()

    print("psnr:{0:.5f}".format(psnr))
    print("SSIM:{0:.5f}".format(ssimloss))