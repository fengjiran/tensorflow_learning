import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.edge_connect import EdgeConnect


if __name__ == '__main__':
    model_path = 'E:\\model\\comparative_models\\edge_connect\\celeba'

    img_dir = 'E:\\model\\experiments\\exp2\\celebahq\\gt_images\\256'

    # Mask or Mask folder
    regular_mask_dir = 'E:\\model\\experiments\\exp2\\mask\\regular_mask'
    irregular_mask_dir = 'E:\\model\\experiments\\exp2\\mask\\irregular_mask'

    # Output dir
    regular_output_dir = 'E:\\model\\experiments\\exp2\\celebahq\\results\\edge-connect\\regular'
    irregular_output_dir = 'E:\\model\\experiments\\exp2\\celebahq\\results\\edge-connect\\irregular'

    # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
    model = 3

    # load config file
    config = Config('config.yml')

    config.MODE = 2
    config.MODEL = model
    config.INPUT_SIZE = 0

    config.TEST_FLIST = img_dir

    config.TEST_MASK_FLIST = regular_mask_dir

    config.RESULTS = regular_output_dir
