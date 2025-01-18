import calibration_module
import cv2
import numpy as np
num_circles=(23, 12)
spacing_size=(160, 160)
pad_pixels=(0, 0)
range_row = (600, 2500)
range_col = (380, 3900)
cali_path = "./cali.bmp"
wrap_path = "./wrap.bmp"
wrap = calibration_module.Calibration(num_circles, spacing_size, pad_pixels)
captured_img = cv2.imread(cali_path)
#b, captured_img, r = cv2.split(captured_img)
wrap_img = cv2.imread(wrap_path)
#b, wrap_img, r = cv2.split(wrap_img)
captured_img_masked = captured_img[range_row[0]:range_row[1], range_col[0]:range_col[1], ...]
wrap_img_masked = wrap_img[range_row[0]:range_row[1], range_col[0]:range_col[1], ...]
wrap.calibrate(captured_img_masked, True)
cv2.imwrite("./1.png", wrap(wrap_img_masked))
