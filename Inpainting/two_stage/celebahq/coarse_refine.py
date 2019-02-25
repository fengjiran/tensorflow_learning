import os
import csv
import platform as pf
import yaml
import numpy as np
import tensorflow as tf
from networks import InpaintingModel
from dataset import Dataset
# from utils import Progbar

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    pass
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        # train_flist = cfg['FLIST_LINUX_7810']
        log_dir = cfg['LOG_DIR_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        pass


class CoarseRefine():
    """Construct model."""

    def __init__(self, config):
        self.cfg = config
        self.model = InpaintingModel(config)
        self.train_dataset = Dataset(config)

    def train(self):
        images, masks, train_iterator = self.train_dataset.load_item()
        flist = self.train_dataset.flist
        total = len(self.train_dataset)
        num_batch = total // self.cfg['BATCH_SIZE']
        max_iteration = self.cfg['MAX_ITERS']

        epoch = 0
        keep_training = True
        step = 0

        # coarse model
        if self.cfg['MODEL'] == 1:
            # train
            coarse_outputs, coarse_outputs_merged, coarse_gen_loss, coarse_dis_loss, coarse_gen_train, coarse_dis_train, logs =\
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

                summary_writer = tf.summary.FileWriter(log_dir)

                sess.run(tf.global_variables_initializer())

                # epoch = 0
                # keep_training = True
                # step = 0
                while keep_training:
                    epoch += 1
                    print('\n\nTraining epoch: %d' % epoch)

                    # progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

                    for i in range(num_batch):
                        _, _, gen_loss, dis_loss = sess.run([coarse_dis_train,
                                                             coarse_gen_train,
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

                        if step >= max_iteration:
                            keep_training = False
                            break

                        logs = [('epoch', epoch), ('iter', step)] + logs

                        progbar.add(images.get_shape().as_list()[0],
                                    values=logs if self.cfg['VERBOSE'] else [x for x in logs if not x[0].startwith('l_')])


if __name__ == '__main__':
    model = CoarseRefine(cfg)
    model.train()
    # print(cfg['GAN_LOSS'] == 'nsgan')
