import os
import csv
import platform as pf
import yaml
import tensorflow as tf
from .dataset import Dataset
from .networks import ColorModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    log_dir = cfg['LOG_DIR_WIN']
    model_dir = cfg['MODEL_PATH_WIN']
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        log_dir = cfg['LOG_DIR_LINUX_7810']
        model_dir = cfg['MODEL_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        log_dir = cfg['LOG_DIR_LINUX_7610']
        model_dir = cfg['MODEL_PATH_LINUX_7610']


class ColorAware():
    """Construct color domain model."""

    def __init__(self, config):
        self.cfg = config
        self.model = ColorModel(config)
        self.dataset = Dataset(config)

    def train(self):
        images, img_masks, img_color_domains = self.dataset.load_items()
        flist = self.dataset.flist
        mask_flist = self.dataset.mask_flist if cfg['MASK'] == 2 else None
        total = len(self.dataset)
        num_batch = total // self.cfg['BATCH_SIZE']
        max_iteration = self.cfg['MAX_ITERS']

        # epoch = 0
        keep_training = True
        # step = 0

        gen_train, dis_train, logs = self.model.build_model(images, img_color_domains, img_masks)
        iterator = self.dataset.train_iterator
        mask_iterator = self.dataset.mask_iterator

        # the saver for model saving and loading
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            iterators = [iterator.initializer, mask_iterator.initializer] if cfg['MASK'] == 2 else iterator.initializer
            feed_dict = {self.dataset.train_filenames: flist,
                         self.dataset.mask_filenames: mask_flist} if cfg['MASK'] == 2 else {self.dataset.train_filenames: flist}
            sess.run(iterators, feed_dict=feed_dict)
            summary_writer = tf.summary.FileWriter(log_dir)

            if self.cfg['firstTimeTrain']:
                step = 0
                epoch = 0
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, os.path.join(model_dir, 'model'))
                step = tf.train.load_variable(os.path.join(model_dir, 'model'), 'gen_global_step')
                epoch = step // num_batch

            with open('logs.csv', 'a+') as f:
                mywrite = csv.writer(f)
                mywrite.writerow(['dis_loss', 'gen_loss', 'gen_gan_loss', 'gen_l1_loss', 'gen_fm_loss'])
            all_summary = tf.summary.merge_all()

            while keep_training:
                epoch += 1
                print('\n\nTraining epoch: %d' % epoch)
                for i in range(num_batch):
                    _, _, logs_ = sess.run([dis_train, gen_train, logs])
                    print('Epoch: {}, Iter: {}'.format(epoch, step))
                    print('-----------dis_loss: {}'.format(logs_[0]))
                    print('-----------gen_loss: {}'.format(logs_[1]))
                    print('-----------gen_gan_loss: {}'.format(logs_[2]))
                    print('-----------gen_l1_loss: {}'.format(logs_[3]))
                    print('-----------gen_fm_loss: {}'.format(logs_[4]))
                    # print('-----------gen_ce_loss: {}'.format(logs_[4]))

                    with open('logs.csv', 'a+') as f:
                        mywrite = csv.writer(f)
                        mywrite.writerow(logs_)

                    if step % self.cfg['SUMMARY_INTERVAL'] == 0:
                        summary = sess.run(all_summary)
                        summary_writer.add_summary(summary, step)

                    if self.cfg['SAVE_INTERVAL'] and step % self.cfg['SAVE_INTERVAL'] == 0:
                        self.model.save(sess, saver, model_dir, 'model')

                    if step >= max_iteration:
                        keep_training = False
                        break

                    step += 1


if __name__ == '__main__':
    model = ColorAware(cfg)
    model.train()
    # images, img_masks, img_color_domains = model.dataset.load_items()
    # color_domains_masked = img_color_domains * (1 - img_masks) + img_masks
    # imgs_masked = images * (1 - img_masks) + img_masks
    # inputs = tf.concat([imgs_masked, color_domains_masked, img_masks * tf.ones_like(images[:, :, :, 0])], axis=3)

    # print(tf.shape(img_color_domains))

    # # x = tf.placeholder(tf.float32, [10, 256, 256, 7])
    # y = model.model.color_domain_generator(img_color_domains)
