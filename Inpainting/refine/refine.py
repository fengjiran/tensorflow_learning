import os
import csv
import platform as pf
import yaml
import tensorflow as tf
from dataset import Dataset
from dataset import MaskDataset
from networks import RefineNet

with open('config_refine_flag.yaml', 'r') as f:
    cfg_flag = yaml.load(f)
    flag = cfg_flag['flag']

if flag == 1:
    cfg_name = 'config_refine_celeba_regular.yaml'
    print('Training refine model with celeba and regular mask')
elif flag == 2:
    cfg_name = 'config_refine_celeba_irregular.yaml'
    print('Training refine model with celeba and irregular mask')
elif flag == 3:
    cfg_name = 'config_refine_celebahq_regular.yaml'
    print('Training refine model with celebahq and regular mask')
elif flag == 4:
    cfg_name = 'config_refine_celebahq_irregular.yaml'
    print('Training refine model with celebahq and irregular mask')
elif flag == 5:
    cfg_name = 'config_refine_psv_regular.yaml'
    print('Training refine model with psv and regular mask')
elif flag == 6:
    cfg_name = 'config_refine_psv_irregular.yaml'
    print('Training refine model with psv and irregular mask')
elif flag == 7:
    cfg_name = 'config_refine_places2_regular.yaml'
    print('Training refine model with places2 and regular mask')
elif flag == 8:
    cfg_name = 'config_refine_places2_irregular.yaml'
    print('Training refine model with places2 and irregular mask')


with open(cfg_name, 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    log_dir = cfg['LOG_DIR_WIN']
    model_dir = cfg['MODEL_PATH_WIN']
    train_flist = cfg['TRAIN_FLIST_WIN']
    val_flist = cfg['VAL_FLIST_WIN']
    test_flist = cfg['TEST_FLIST_WIN']
    mask_flist = cfg['MASK_FLIST_WIN']
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        log_dir = cfg['LOG_DIR_LINUX_7810']
        model_dir = cfg['MODEL_PATH_LINUX_7810']
        train_flist = cfg['TRAIN_FLIST_LINUX_7810']
        val_flist = cfg['VAL_FLIST_LINUX_7810']
        test_flist = cfg['TEST_FLIST_LINUX_7810']
        mask_flist = cfg['MASK_FLIST_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        log_dir = cfg['LOG_DIR_LINUX_7610']
        model_dir = cfg['MODEL_PATH_LINUX_7610']
        train_flist = cfg['TRAIN_FLIST_LINUX_7610']
        val_flist = cfg['VAL_FLIST_LINUX_7610']
        test_flist = cfg['TEST_FLIST_LINUX_7610']
        mask_flist = cfg['MASK_FLIST_LINUX_7610']


class RefineModel():
    """Construct refine model."""

    def __init__(self, config):
        self.cfg = config
        self.model = RefineNet(config)
        self.train_dataset = Dataset(config, train_flist)
        self.val_dataset = Dataset(config, val_flist)
        self.mask_dataset = MaskDataset(config, mask_flist)

    def train(self):
        images, img_grays, edges, img_color_domains = self.train_dataset.load_items()
        val_images, val_grays, val_edges, val_img_color_domains = self.val_dataset.load_items()
        img_masks = self.mask_dataset.load_items()

        total = len(self.train_dataset)
        num_batch = total // self.cfg['BATCH_SIZE']
        max_iteration = self.cfg['MAX_ITERS']

        keep_training = True
