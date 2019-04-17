import os
import csv
import platform as pf
import yaml
import tensorflow as tf
from dataset import Dataset
from dataset import MaskDataset
from .networks import InpaintModel

with open('config.yaml', 'r') as f:
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


class JointModel():
    """Construct joint model."""

    def __init__(self, config):
        self.cfg = config
        self.model = InpaintModel(config)

        self.train_dataset = Dataset(config, train_flist)
        self.val_dataset = Dataset(config, val_flist)
        self.mask_dataset = MaskDataset(config, mask_flist)

    def pre_train(self):
        images, edges, img_color_domains = self.train_dataset.load_items()
        val_images, val_edges, val_img_color_domains = self.val_dataset.load_items()
        img_masks = self.mask_dataset.load_items()

        total = len(self.train_dataset)
        num_batch = total // self.cfg['BATCH_SIZE']
        max_iteration = self.cfg['MAX_ITERS']

        keep_training = True

        gen_train, dis_train, logs = self.model.build_model(images, edges, img_color_domains, img_masks)
        val_logs = self.model.eval_model(val_images, val_edges, val_img_color_domains, img_masks)
