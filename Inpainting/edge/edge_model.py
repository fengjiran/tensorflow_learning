import csv
import platform as pf
import yaml
import tensorflow as tf
from dataset import Dataset
from networks import EdgeModel

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    pass
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        log_dir = cfg['LOG_DIR_LINUX_7810']
        model_dir = cfg['MODEL_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        pass


class EdgeAware():
    """Construct edge model."""

    def __init__(self, config):
        self.cfg = config
        self.model = EdgeModel(config)
        self.dataset = Dataset(config)

    def train(self):
        images, img_grays, img_edges, img_masks = self.dataset.load_items()
        flist = self.dataset.flist
        mask_flist = self.dataset.mask_flist if cfg['MASK'] == 2 else None
        total = len(self.dataset)
        num_batch = total // self.cfg['BATCH_SIZE']
        max_iteration = self.cfg['MAX_ITERS']

        epoch = 0
        keep_training = True
        step = 0

        gen_train, dis_train, logs = self.model.build_model(img_grays, img_edges, img_masks)
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
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(log_dir)

            with open('logs.csv', 'a+') as f:
                mywrite = csv.writer(f)
                mywrite.writerow(['dis_loss', 'gen_loss', 'gen_gan_loss', 'gen_fm_loss'])
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
    model = EdgeAware(cfg)
    model.train()
