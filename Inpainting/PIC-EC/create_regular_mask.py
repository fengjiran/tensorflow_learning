import os
import numpy as np
from imageio import imwrite

smaple_dir = 'E:\\model\\experiments\\exp2\\celebahq\\mask\\regular_mask'


def generate_regular_mask(mask_size=256, hole_size=128):
    top = np.random.randint(0, hole_size + 1)
    left = np.random.randint(0, hole_size + 1)
    mask = np.pad(array=np.zeros((hole_size, hole_size)),
                  pad_width=((top, mask_size - hole_size - top),
                             (left, mask_size - hole_size - left)),
                  mode='constant',
                  constant_values=1)
    return mask


if __name__ == '__main__':
    for i in range(100):
        mask = generate_regular_mask()
        imwrite(os.path.join(smaple_dir, 'regular_mask_%02d.png' % (i + 1)), mask)
