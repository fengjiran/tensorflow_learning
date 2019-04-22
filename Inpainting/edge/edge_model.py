import os
import csv
import platform as pf
import yaml
import tensorflow as tf
from dataset import Dataset
from dataset import MaskDataset
from networks import EdgeModel

with open('config_edge_flag.yaml', 'r') as f:
    cfg_flag = yaml.load(f)
    flag = cfg_flag['flag']

if flag == 1:
    cfg_name = 'config_edge_celeba_regular.yaml'
elif flag == 2:
    cfg_name = 'config_edge_celeba_irregular.yaml'
elif flag == 3:
    cfg_name = 'config_edge_celebahq_regular.yaml'
elif flag == 4:
    cfg_name = 'config_edge_celebahq_irregular.yaml'
elif flag == 5:
    cfg_name = 'config_edge_psv_regular.yaml'
elif flag == 6:
    cfg_name = 'config_edge_psv_irregular.yaml'
elif flag == 7:
    cfg_name = 'config_edge_places2_regular.yaml'
elif flag == 8:
    cfg_name = 'config_edge_places2_irregular.yaml'

with open(cfg_name, 'r') as f:
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


class EdgeAware():
    """Construct edge model."""

    def __init__(self, config):
        self.cfg = config
        self.model = EdgeModel(config)

        self.train_dataset = Dataset(config, train_flist)
        self.val_dataset = Dataset(config, val_flist)
        self.mask_dataset = MaskDataset(config, mask_flist)

    def train(self):
        images, img_grays, img_edges = self.train_dataset.load_items()
        val_images, val_img_grays, val_img_edges = self.val_dataset.load_items()
        img_masks = self.mask_dataset.load_items()

        total = len(self.train_dataset)
        num_batch = total // self.cfg['BATCH_SIZE']
        max_iteration = self.cfg['MAX_ITERS']

        keep_training = True

        gen_train, dis_train, logs = self.model.build_model(img_grays, img_edges, img_masks)
        val_logs = self.model.eval_model(val_img_grays, val_img_edges, img_masks)

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

            if self.cfg['firstTimeTrain']:
                step = 0
                epoch = 0
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, os.path.join(model_dir, 'model'))
                step = tf.train.load_variable(os.path.join(model_dir, 'model'), 'gen_global_step')
                epoch = step // num_batch - 1

            with open(os.path.join(log_dir, 'logs.csv'), 'a+') as f:
                mywrite = csv.writer(f)
                mywrite.writerow(['dis_loss', 'gen_loss', 'gen_gan_loss', 'gen_fm_loss'])

            summary_writer = tf.summary.FileWriter(log_dir)
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
                    print('-----------gen_fm_loss: {}'.format(logs_[3]))
                    print('-----------gen_ce_loss: {}'.format(logs_[4]))

                    with open(os.path.join(log_dir, 'logs.csv'), 'a+') as f:
                        mywrite = csv.writer(f)
                        mywrite.writerow(logs_)

                    if step % self.cfg['EVAL_INTERVAL'] == 0:
                        val_logs_ = sess.run(val_logs)
                        print('-----------val_precision: {}'.format(val_logs_[0]))
                        print('-----------val_recall: {}'.format(val_logs_[1]))

                        summary = sess.run(all_summary)
                        summary_writer.add_summary(summary, step)

                    if self.cfg['SAVE_INTERVAL'] and step % self.cfg['SAVE_INTERVAL'] == 0:
                        self.model.save(sess, saver, model_dir, 'model')

                    if step >= max_iteration:
                        keep_training = False
                        break

                    step += 1


if __name__ == '__main__':
    model = EdgeAware(cfg)
    model.train()
