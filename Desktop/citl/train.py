import props.prop_model
import dataload
from torch import optim
from trainer_kit import *
from torch import nn
import utils
import cali.calibration_module
num_circles=(47, 26)
range_row = (600, 2500)#TODO annotation
range_col = (380, 3900)#

#硬件相关设置及传播距离设置
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
wavelength = 520 * nm
feature_size = (3.6 * um, 3.6 * um)
F_aperture = 1.0
slm_res = (2160, 3840)
roi_res = (2160, 3840)
slm_data_path = "./data/cgh"
img_data_path = "./data/img"
calibrate_img_path = "./data/forcali/cali.bmp"
device = "cuda:0"
#device = "cpu"
device_ids = [0, 1]  
lr = 4e-4
epochs = 5
prop_dist = 1 * mm
prop_dists = [0 * mm, 1 * mm] 
prop_dists_from_wrp = [p - prop_dist for p in prop_dists]
model = props.prop_model.CNNpropCNN(prop_dist=prop_dist, 
                                    wavelength=wavelength,
                                    feature_size=feature_size,
                                    prop_dists_from_wrp=prop_dists_from_wrp, 
                                    use_wt= True)
#初始化数据集
distindex = {0:0, 1:1}
citl_set = dataload.CaliDataset(channel=1, depths=len(distindex),distindex = distindex,slm_data_path=slm_data_path, img_data_path=img_data_path)
Loss = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr= lr)
calibrate = cali.calibration_module.Calibration(num_circles)
#model.load_state_dict(torch.load(""))
if __name__ == "__main__":
    model.to(device)
    #model = nn.DataParallel(model, device_ids=device_ids)
    calibrate_img = cv2.imread(calibrate_img_path)
    #captured_img_masked = calibrate_img[range_row[0]:range_row[1], range_col[0]:range_col[1], ...]
    #calibrate.calibrate(captured_img_masked)
    for epoch in range(epochs):
        for cgh, img in citl_set:
            optimizer.zero_grad()
            cgh, img = cgh.to(device)/255.0, img.to(device)/255.0
            #img_masked = img[range_row[0]:range_row[1], range_col[0]:range_col[1], ...]
            #img = calibrate(img)
            output = model(cgh).squeeze().abs() #
            #output_amps = utils.crop_image(output, roi_res, stacked_complex=False)
            loss = Loss(output, img)#
            loss.backward()
            optimizer.step() #
    torch.save(model.state_dict(), "./out/model.pt")



    

