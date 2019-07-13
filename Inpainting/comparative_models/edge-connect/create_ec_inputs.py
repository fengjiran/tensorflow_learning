import os
from imageio import imread
from imageio import imwrite
import numpy as np
import cv2

img_path = 'E:\\model\\experiments\\exp2\\celebahq\\gt_images\\256'
irregular_mask_path = 'E:\\model\\experiments\\exp2\\mask\\irregular_mask'
regular_mask_path = 'E:\\model\\experiments\\exp2\\mask\\regular_mask'

irregular_masked_img_path = 'E:\\model\\experiments\\exp2\\celebahq\\results\\edge-connect\\input_images\\irregular'
regular_masked_img_path = 'E:\\model\\experiments\\exp2\\celebahq\\results\\edge-connect\\input_images\\irregular'

img_dir = os.listdir(img_path)
irregular_mask_dir = os.listdir(irregular_mask_path)
regular_mask_dir = os.listdir(regular_mask_path)
