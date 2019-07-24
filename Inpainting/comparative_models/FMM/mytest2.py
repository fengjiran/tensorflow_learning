import os
from imageio import imread
from imageio import imwrite
import cv2

celebahq_img_dir = 'E:\\model\\experiments\\exp2\\celebahq\\gt_images\\256'
psv_img_dir = 'E:\\model\\experiments\\exp2\\psv\\gt_images'

regular_mask_dir = 'E:\\model\\experiments\\exp2\\mask\\regular_mask'
irregular_mask_dir = 'E:\\model\\experiments\\exp2\\mask\\irregular_mask'

irregular_celebahq_results_dir = 'E:\\model\\experiments\\exp2\\celebahq\\results\\FMM\\irregular'
regular_celebahq_results_dir = 'E:\\model\\experiments\\exp2\\celebahq\\results\\FMM\\regular'

irregular_psv_results_dir = 'E:\\model\\experiments\\exp2\\psv\\results\\FMM\\irregular'
regular_psv_results_dir = 'E:\\model\\experiments\\exp2\\psv\\results\\FMM\\regular'

# for celebahq
celebahq_flist = os.listdir(celebahq_img_dir)
regular_mask_flist = os.listdir(regular_mask_dir)
irregular_mask_flist = os.listdir(irregular_mask_dir)

i = 1
for path1, path2 in zip(celebahq_flist, regular_mask_flist):
    img_path = os.path.join(celebahq_img_dir, path1)
    mask_path = os.path.join(regular_mask_dir, path2)

    img = imread(img_path)
    mask = imread(mask_path)
    mask = 255 - mask
    dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    imwrite(os.path.join(regular_celebahq_results_dir, 'celebahq_regular_result_%03d.png' % i), dst)

    i += 1


i = 1
for path1, path2 in zip(celebahq_flist, irregular_mask_flist):
    img_path = os.path.join(celebahq_img_dir, path1)
    mask_path = os.path.join(irregular_mask_dir, path2)

    img = imread(img_path)
    mask = imread(mask_path)
    mask = 255 - mask
    dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    imwrite(os.path.join(irregular_celebahq_results_dir, 'celebahq_irregular_result_%03d.png' % i), dst)

    i += 1

# for psv
psv_flist = os.listdir(psv_img_dir)

i = 1
for path1, path2 in zip(psv_flist, regular_mask_flist):
    img_path = os.path.join(psv_img_dir, path1)
    mask_path = os.path.join(regular_mask_dir, path2)

    img = imread(img_path)
    mask = imread(mask_path)
    mask = 255 - mask
    dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    imwrite(os.path.join(regular_psv_results_dir, 'psv_regular_result_%03d.png' % i), dst)

    i += 1

i = 1
for path1, path2 in zip(psv_flist, irregular_mask_flist):
    img_path = os.path.join(psv_img_dir, path1)
    mask_path = os.path.join(irregular_mask_dir, path2)

    img = imread(img_path)
    mask = imread(mask_path)
    mask = 255 - mask
    dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    imwrite(os.path.join(irregular_psv_results_dir, 'psv_irregular_result_%03d.png' % i), dst)

    i += 1
