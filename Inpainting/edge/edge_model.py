import csv
import platform as pf
import yaml
import tensorflow as tf
from .dataset import Dataset
from .networks import EdgeModel

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
        images, img_grays, img_edges, img_masks = self.dataset.load_items()
        flist = self.dataset.flist
        total = len(self.dataset)
        num_batch = total // self.cfg['BATCH_SIZE']
        max_iteration = self.cfg['MAX_ITERS']

        epoch = 0
        keep_training = True
        step = 0

        gen_train, dis_train, logs = self.model.build_model(img_grays, img_edges, img_masks)

        # the saver for model saving and loading
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            pass
