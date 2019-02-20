import os
import numpy as np
import tensorflow as tf
from .networks import InpaintingModel
from .dataset import Dataset


class CoarseRefine():
    """Construct model."""

    def __init__(self, config):
        self.cfg = config
        self.model = InpaintingModel(config)
        self.train_dataset = Dataset(config)

    def train(self):
        images, masks, train_iterator = self.train_dataset.load_item()
        flist = self.train_dataset.load_flist(self.cfg['FLIST'])
        num_batch = len(flist) // self.cfg['BATCH_SIZE']

        coarse_outputs, coarse_outputs_merged, coarse_gen_loss, coarse_dis_loss, coarse_gen_train, coarse_dis_train =\
            self.model.build_coarse_model(images, masks)

        coarse_dis_train_ops = []
        for i in range(5):
            coarse_dis_train_ops.append(coarse_dis_train)
        coarse_dis_train = tf.group(*coarse_dis_train_ops)

        all_summary = tf.summary.merge_all()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(train_iterator.initializer, feed_dict={self.train_dataset.train_filenames: flist})

            summary_writer = tf.summary.FileWriter(logdir, sess.graph)

            step = 0
            while True:
                _, _, gen_loss, dis_loss = sess.run([coarse_gen_train,
                                                     coarse_dis_loss,
                                                     coarse_gen_loss,
                                                     coarse_dis_loss])
                print('Epoch: {}, Iter: {}, coarse_gen_loss: {}, coarse_dis_loss: {}'.format(
                    step // num_batch + 1,
                    step,
                    gen_loss,
                    dis_loss
                ))

                if step % 200 == 0:
                    summary = sess.run(all_summary)
                    summary_writer.add_summary(summary, step)

                step += 1
