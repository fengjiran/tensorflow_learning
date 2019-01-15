import os
import numpy as np


class Dataset():
    """Construct dataset class."""

    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        self.augment = augment
        self.training = training
