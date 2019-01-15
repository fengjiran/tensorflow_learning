import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
