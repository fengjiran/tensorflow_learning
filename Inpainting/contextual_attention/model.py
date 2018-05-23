from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils import spatial_discounting_mask
from utils import random_bbox
from utils import bbox2mask
from utils import local_patch
from utils import gan_wgan_loss
