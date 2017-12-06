from __future__ import print_function

import platform
import numpy as np
import tensorflow as tf

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\imagenet_train_path_win.pickle'
    model_path = 'E:\\TensorFlow_Learning\\Inpainting\\globally_locally\\models_without_adv_l1'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/imagenet_train_path_linux.pickle'
    model_path = '/home/richard/TensorFlow_Learning/Inpainting/globally_locally/models_without_adv_l1'


g1 = tf.Graph()

print('tf.get_default_graph()=', tf.get_default_graph())
print('g1                    =', g1)

with g1.as_default():
    with tf.Session(graph=g1) as sess1:
        pass
