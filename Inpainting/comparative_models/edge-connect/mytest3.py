import os
import random
import time
import cv2
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.edge_connect import EdgeConnect


if __name__ == '__main__':
    model_path = 'E:\\model\\comparative_models\\edge_connect\\psv'
    # model_path = 'E:\\model\\comparative_models\\edge_connect\\celeba'

    img_dir = 'E:\\model\\experiments\\exp3\\psv\\gt_images'
    # img_dir = 'E:\\model\\experiments\\exp3\\celebahq\\gt_images'

    # Mask or Mask folder
    regular_mask_dir = 'E:\\model\\experiments\\exp3\\psv\\mask\\128'
    # irregular_mask_dir = 'E:\\model\\experiments\\exp2\\mask\\irregular_mask'

    # Output dir
    # regular_output_dir = 'E:\\model\\experiments\\exp2\\psv\\results\\edge-connect\\regular'
    # irregular_output_dir = 'E:\\model\\experiments\\exp2\\psv\\results\\edge-connect\\irregular'

    regular_output_dir = 'E:\\model\\experiments\\exp3\\psv\\results\\edge-connect\\128'
    # irregular_output_dir = 'E:\\model\\experiments\\exp2\\celebahq\\results\\edge-connect\\irregular'

    # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
    model = 3

    # load config file
    config = Config('config.yml')

    config.MODE = 2
    config.MODEL = model
    config.INPUT_SIZE = 0

    config.PATH = model_path

    config.TEST_FLIST = img_dir

    config.TEST_MASK_FLIST = regular_mask_dir

    config.RESULTS = regular_output_dir

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    config.DEVICE = torch.device("cuda")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = EdgeConnect(config)
    model.load()

    print('\nstart testing...\n')
    start = time.time()
    model.test()
    end = time.time()
    print((end - start) * 1000)
