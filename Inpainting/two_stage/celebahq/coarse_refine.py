import os
import csv
import platform as pf
import yaml
import numpy as np
import tensorflow as tf
from networks import InpaintingModel
from dataset import Dataset
from utils import create_mask

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    pass
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        # train_flist = cfg['FLIST_LINUX_7810']
        log_dir = cfg['LOG_DIR_LINUX_7810']
        model_dir = cfg['MODEL_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        pass


class CoarseRefine():
    """Construct model."""

    def __init__(self, config):
        self.cfg = config
        self.model = InpaintingModel(config)
        self.train_dataset = Dataset(config)

    def train(self):
        images, train_iterator = self.train_dataset.load_item()
        masks = create_mask(self.cfg['INPUT_SIZE'], self.cfg['INPUT_SIZE'],
                            self.cfg['INPUT_SIZE'] // 2, self.cfg['INPUT_SIZE'] // 2)

        flist = self.train_dataset.flist
        total = len(self.train_dataset)
        num_batch = total // self.cfg['BATCH_SIZE']
        max_iteration = self.cfg['MAX_ITERS']

        epoch = 0
        keep_training = True
        step = 0

        # temp = self.model.test(images, masks)

        coarse_returned, refine_returned, joint_returned = self.model.build_model(images, masks)

        coarse_outputs, coarse_outputs_merged, coarse_gen_train, coarse_dis_train, coarse_logs = coarse_returned

        refine_outputs, refine_outputs_merged, refine_gen_train, refine_dis_train, refine_logs = refine_returned

        joint_outputs, joint_outputs_merged, joint_gen_train, joint_dis_train, joint_logs = joint_returned

        # the saver for model saving and loading
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(train_iterator.initializer, feed_dict={self.train_dataset.train_filenames: flist})
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(log_dir)

            # coarse model
            if self.cfg['MODEL'] == 1:
                # train

                with open('coarse_logs.csv', 'a+') as f:
                    mywrite = csv.writer(f)
                    mywrite.writerow(['dis_loss',
                                      'gen_loss',
                                      'gen_gan_loss',
                                      'gen_l1_loss',
                                      'gen_style_loss',
                                      'gen_content_loss'])

                coarse_summary_collection = [tf.get_collection(tf.GraphKeys.SUMMARIES, 'coarse_dis_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'coarse_gen_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'coarse_gen_gan_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'coarse_gen_l1_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'coarse_gen_style_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'coarse_gen_content_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'coarse_gt_masked_inpainted')]
                coarse_summary = tf.summary.merge(coarse_summary_collection)
                # all_summary = tf.summary.merge_all()

                # epoch = 0
                # keep_training = True
                # step = 0
                while keep_training:
                    epoch += 1
                    print('\n\nTraining epoch: %d' % epoch)

                    for i in range(num_batch):
                        _, _, coarse_logs_ = sess.run([coarse_dis_train,
                                                       coarse_gen_train,
                                                       coarse_logs])
                        print('Epoch: {}, Iter: {}'.format(epoch, step))
                        print('-----------dis_loss: {}'.format(coarse_logs_[0]))
                        print('-----------gen_loss: {}'.format(coarse_logs_[1]))
                        print('-----------gen_gan_loss: {}'.format(coarse_logs_[2]))
                        print('-----------gen_l1_loss: {}'.format(coarse_logs_[3]))
                        print('-----------gen_style_loss: {}'.format(coarse_logs_[4]))
                        print('-----------gen_content_loss: {}'.format(coarse_logs_[5]))

                        with open('coarse_logs.csv', 'a+') as f:
                            mywrite = csv.writer(f)
                            mywrite.writerow(coarse_logs_)

                        if step % self.cfg['SUMMARY_INTERVAL'] == 0:
                            summary = sess.run(coarse_summary)
                            summary_writer.add_summary(summary, step)

                        if step % self.cfg['SAVE_INTERVAL'] == 0:
                            self.model.save(sess, saver, model_dir, 'model')

                        if step >= max_iteration:
                            keep_training = False
                            break

                        step += 1

            # refine model
            elif self.cfg['MODEL'] == 2:
                # train
                with open('refine_logs.csv', 'a+') as f:
                    mywrite = csv.writer(f)
                    mywrite.writerow(['dis_loss',
                                      'gen_loss',
                                      'gen_gan_loss',
                                      'gen_l1_loss',
                                      'gen_style_loss',
                                      'gen_content_loss'])

                refine_summary_collection = [tf.get_collection(tf.GraphKeys.SUMMARIES, 'refine_dis_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'refine_gen_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'refine_gen_gan_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'refine_gen_l1_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'refine_gen_style_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'refine_gen_content_loss'),
                                             tf.get_collection(tf.GraphKeys.SUMMARIES, 'refine_gt_masked_coarse_refine')]
                refine_summary = tf.summary.merge(coarse_summary_collection)


if __name__ == '__main__':
    model = CoarseRefine(cfg)
    model.train()
    # print(cfg['GAN_LOSS'] == 'nsgan')
