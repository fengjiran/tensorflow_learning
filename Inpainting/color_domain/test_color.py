import os
import yaml
import platform as pf
import numpy as np
import tensorflow as tf
import cv2
from .networks import ColorModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)
