import csv
import platform as pf
import yaml
import tensorflow as tf
from dataset import Dataset
from networks import EdgeModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    pass
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        log_dir = cfg['LOG_DIR_LINUX_7810']
        model_dir = cfg['MODEL_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        pass


class Edge():
    """Construct edge model."""

    def __init__(self, config):
        self.cfg = config
        self.model = EdgeModel(config)
        self.dataset = Dataset(config)

    def train(self):
        pass
