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
