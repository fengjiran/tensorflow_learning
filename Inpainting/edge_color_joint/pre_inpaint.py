import os
import csv
import platform as pf
import yaml
import tensorflow as tf
from .dataset import Dataset
from .networks import InpaintModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    log_dir = cfg['LOG_DIR_WIN']
    model_dir = cfg['MODEL_PATH_WIN']
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        log_dir = cfg['LOG_DIR_LINUX_7810']
        model_dir = cfg['MODEL_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        log_dir = cfg['LOG_DIR_LINUX_7610']
        model_dir = cfg['MODEL_PATH_LINUX_7610']


class PreInpaint():
    """Construct pre inpaint model."""

    def __init__(self, config):
        self.cfg = config
        self.model = InpaintModel(config)
        self.dataset = Dataset(config)

    def train(self):
        images, img_grays, img_edges, img_masks, img_color_domains = self.dataset.load_items()
        flist = self.dataset.flist  # for image
        mask_flist = self.dataset.mask_flist if self.cfg['MASK'] == 2 else None
        total = len(self.dataset)
        num_batch = total // self.cfg['BATCH_SIZE']

        max_iteration = self.cfg['MAX_ITERS']

        # epoch = 0
        keep_training = True
        # step = 0

        gen_train, dis_train, logs = self.model.build_model(img_grays, img_edges, img_masks)
        iterator = self.dataset.train_iterator
        mask_iterator = self.dataset.mask_iterator
