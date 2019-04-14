import os
import csv
import platform as pf
import yaml
import tensorflow as tf
from dataset import Dataset
from dataset import MaskDataset
from networks import ColorModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    log_dir = cfg['LOG_DIR_WIN']
    model_dir = cfg['MODEL_PATH_WIN']
    train_flist = cfg['TRAIN_FLIST_WIN']
    val_flist = cfg['VAL_FLIST_WIN']
    test_flist = cfg['TEST_FLIST_WIN']
    mask_flist = cfg['MASK_FLIST_WIN']
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        log_dir = cfg['LOG_DIR_LINUX_7810']
        model_dir = cfg['MODEL_PATH_LINUX_7810']
        train_flist = cfg['TRAIN_FLIST_LINUX_7810']
        val_flist = cfg['VAL_FLIST_LINUX_7810']
        test_flist = cfg['TEST_FLIST_LINUX_7810']
        mask_flist = cfg['MASK_FLIST_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        log_dir = cfg['LOG_DIR_LINUX_7610']
        model_dir = cfg['MODEL_PATH_LINUX_7610']
        train_flist = cfg['TRAIN_FLIST_LINUX_7610']
        val_flist = cfg['VAL_FLIST_LINUX_7610']
        test_flist = cfg['TEST_FLIST_LINUX_7610']
        mask_flist = cfg['MASK_FLIST_LINUX_7610']


class ColorAware():
    """Construct color domain model."""

    def __init__(self, config):
        self.cfg = config
        self.model = ColorModel(config)

        self.train_dataset = Dataset(config, train_flist)
        self.val_dataset = Dataset(config, val_flist)
        self.mask_dataset = MaskDataset(config, mask_flist)

    def train(self):
        images, img_color_domains = self.train_dataset.load_items()
        val_images, val_img_color_domains = self.val_dataset.load_items()
        img_masks = self.mask_dataset.load_items()

        total = len(self.train_dataset)
        num_batch = total // self.cfg['BATCH_SIZE']
        max_iteration = self.cfg['MAX_ITERS']

        keep_training = True

        gen_train, dis_train, logs = self.model.build_model(images, img_color_domains, img_masks)
        val_logs = self.model.eval_model(val_images, val_img_color_domains, img_masks)

        train_iterator = self.train_dataset.iterator
        val_iterator = self.val_dataset.iterator
        mask_iterator = self.mask_dataset.mask_iterator

        # the saver for model saving and loading
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if cfg['MASK'] == 2:
                iterators = [train_iterator.initializer, val_iterator.initializer, mask_iterator.initializer]
            else:
                iterators = [train_iterator.initializer, val_iterator.initializer]

            feed_dict = {self.train_dataset.filenames: self.train_dataset.flist,
                         self.val_dataset.filenames: self.val_dataset.flist}

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
                mywrite.writerow(['dis_loss', 'gen_loss', 'gen_gan_loss', 'gen_l1_loss', 'gen_fm_loss',
                                  'psnr', 'ssim', 'l1', 'l2'])
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

                    if step % self.cfg['EVAL_INTERVAL'] == 0:
                        val_logs_ = sess.run(val_logs)
                        print('-----------psnr: {}'.format(val_logs_[0]))
                        print('-----------ssim: {}'.format(val_logs_[1]))
                        print('-----------l1: {}'.format(val_logs_[2]))
                        print('-----------l2: {}'.format(val_logs_[3]))

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
    # model.train()

    model.test_eval()

    # images, img_masks, img_color_domains = model.dataset.load_items()
    # color_domains_masked = img_color_domains * (1 - img_masks) + img_masks
    # imgs_masked = images * (1 - img_masks) + img_masks
    # inputs = tf.concat([imgs_masked, color_domains_masked, img_masks * tf.ones_like(images[:, :, :, 0])], axis=3)

    # print(tf.shape(img_color_domains))

    # # x = tf.placeholder(tf.float32, [10, 256, 256, 7])
    # y = model.model.color_domain_generator(img_color_domains)
