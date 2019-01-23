import os
import numpy as np
import tensorflow as tf
from .networks import InpaintingModel
from.dataset import Dataset


class CoarseRefine():
    """Construct model."""

    def __init__(self, config):
        self.cfg = config
        self.model = InpaintingModel(config)
        self.train_dataset = Dataset(config)
        images, masks = self.train_dataset.load_item()
        flist = self.train_dataset.load_flist(self.cfg['FLIST'])

        coarse_outputs, coarse_outputs_merged, coarse_gen_loss, coarse_dis_loss, coarse_gen_train, coarse_dis_train =\
            self.model.build_coarse_model(images, masks)

        coarse_dis_train_ops = []
        for i in range(5):
            coarse_dis_train_ops.append(coarse_dis_train)
        coarse_dis_train = tf.group(*coarse_dis_train_ops)

    def train(self):
        pass
