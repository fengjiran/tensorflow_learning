from __future__ import print_function

import argparse
import cv2
import numpy as np
import tensorflow as tf

from model import CompletionModel

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')

args = parser.parse_args()
model = CompletionModel()
image = cv2.imread(args.image)
mask = cv2.imread(args.mask)
