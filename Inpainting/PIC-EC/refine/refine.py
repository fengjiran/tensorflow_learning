import os
import csv
import platform as pf
import yaml
import tensorflow as tf
from dataset import Dataset
from dataset import MaskDataset
from networks import RefineNet

with open('config_refine_flag.yaml', 'r') as f:
    cfg_flag = yaml.load(f)
    flag = cfg_flag['flag']

if flag == 1:
    cfg_name = 'config_refine_celeba_regular.yaml'
    print('Training refine model with celeba and regular mask')
elif flag == 2:
    cfg_name = 'config_refine_celeba_irregular.yaml'
    print('Training refine model with celeba and irregular mask')
elif flag == 3:
    cfg_name = 'config_refine_celebahq_regular.yaml'
    print('Training refine model with celebahq and regular mask')
elif flag == 4:
    cfg_name = 'config_refine_celebahq_irregular.yaml'
    print('Training refine model with celebahq and irregular mask')
elif flag == 5:
    cfg_name = 'config_refine_psv_regular.yaml'
    print('Training refine model with psv and regular mask')
elif flag == 6:
    cfg_name = 'config_refine_psv_irregular.yaml'
    print('Training refine model with psv and irregular mask')
elif flag == 7:
    cfg_name = 'config_refine_places2_regular.yaml'
    print('Training refine model with places2 and regular mask')
elif flag == 8:
    cfg_name = 'config_refine_places2_irregular.yaml'
    print('Training refine model with places2 and irregular mask')


with open(cfg_name, 'r') as f:
    cfg = yaml.load(f)

if pf.system() == 'Windows':
    log_dir = cfg['LOG_DIR_WIN']
    model_dir = cfg['MODEL_PATH_WIN']
    train_flist = cfg['TRAIN_FLIST_WIN']
    val_flist = cfg['VAL_FLIST_WIN']
    test_flist = cfg['TEST_FLIST_WIN']
    mask_flist = cfg['MASK_FLIST_WIN']
    edge_model_dir = cfg['EDGE_PATH_WIN']
    color_model_dir = cfg['COLOR_PATH_WIN']
    joint_model_dir = cfg['JOINT_PATH_WIN']
elif pf.system() == 'Linux':
    if pf.node() == 'icie-Precision-Tower-7810':
        log_dir = cfg['LOG_DIR_LINUX_7810']
        model_dir = cfg['MODEL_PATH_LINUX_7810']
        train_flist = cfg['TRAIN_FLIST_LINUX_7810']
        val_flist = cfg['VAL_FLIST_LINUX_7810']
        test_flist = cfg['TEST_FLIST_LINUX_7810']
        mask_flist = cfg['MASK_FLIST_LINUX_7810']
        edge_model_dir = cfg['EDGE_PATH_LINUX_7810']
        color_model_dir = cfg['COLOR_PATH_LINUX_7810']
        joint_model_dir = cfg['JOINT_PATH_LINUX_7810']
    elif pf.node() == 'icie-Precision-T7610':
        log_dir = cfg['LOG_DIR_LINUX_7610']
        model_dir = cfg['MODEL_PATH_LINUX_7610']
        train_flist = cfg['TRAIN_FLIST_LINUX_7610']
        val_flist = cfg['VAL_FLIST_LINUX_7610']
        test_flist = cfg['TEST_FLIST_LINUX_7610']
        mask_flist = cfg['MASK_FLIST_LINUX_7610']
        edge_model_dir = cfg['EDGE_PATH_LINUX_7610']
        color_model_dir = cfg['COLOR_PATH_LINUX_7610']
        joint_model_dir = cfg['JOINT_PATH_LINUX_7610']


class RefineModel():
    """Construct refine model."""

    def __init__(self, config):
        self.cfg = config
        self.model = RefineNet(config)
        self.train_dataset = Dataset(config, train_flist)
        self.val_dataset = Dataset(config, val_flist)
        self.mask_dataset = MaskDataset(config, mask_flist)

    def train(self):
        images, img_grays, edges, img_color_domains = self.train_dataset.load_items()
        val_images, val_grays, val_edges, val_img_color_domains = self.val_dataset.load_items()
        img_masks = self.mask_dataset.load_items()

        total = len(self.train_dataset)
        num_batch = total // self.cfg['BATCH_SIZE']
        max_iteration = self.cfg['MAX_ITERS']

        keep_training = True

        gen_train, dis_train, logs = self.model.build_model(images, img_grays, edges, img_color_domains, img_masks)
        val_logs = self.model.eval_model(val_images, val_grays, val_edges, val_img_color_domains, img_masks)

        train_iterator = self.train_dataset.iterator
        val_iterator = self.val_dataset.iterator
        mask_iterator = self.mask_dataset.mask_iterator

        # get model variables
        edge_gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'edge_generator')
        color_gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'color_generator')
        inpaint_gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inpaint_generator')
        dis_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inpaint_discriminator')

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'vgg')

        for var in vgg_var_list:
            var_list.remove(var)

        # the saver for model saving and loading
        saver = tf.train.Saver(var_list)

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

                edge_assign_ops = []
                color_assign_ops = []
                inpaint_assign_ops = []
                dis_assign_ops = []

                for var in edge_gen_vars:
                    vname = var.name
                    from_name = vname
                    var_value = tf.train.load_variable(os.path.join(edge_model_dir, 'model'), from_name)
                    edge_assign_ops.append(tf.assign(var, var_value))
                sess.run(edge_assign_ops)
                print('Edge model loaded!')

                for var in color_gen_vars:
                    vname = var.name
                    from_name = vname
                    var_value = tf.train.load_variable(os.path.join(color_model_dir, 'model'), from_name)
                    color_assign_ops.append(tf.assign(var, var_value))
                sess.run(color_assign_ops)
                print('Color model loaded!')

                for var in inpaint_gen_vars:
                    vname = var.name
                    from_name = vname
                    var_value = tf.train.load_variable(os.path.join(joint_model_dir, 'model'), from_name)
                    inpaint_assign_ops.append(tf.assign(var, var_value))
                sess.run(inpaint_assign_ops)
                print('Joint model loaded!')

                for var in dis_vars:
                    vname = var.name
                    from_name = vname
                    var_value = tf.train.load_variable(os.path.join(joint_model_dir, 'model'), from_name)
                    dis_assign_ops.append(tf.assign(var, var_value))
                sess.run(dis_assign_ops)
                print('Dis model loaded!')
            else:
                sess.run(tf.global_variables_initializer())
                # # add below
                # edge_assign_ops = []
                # color_assign_ops = []
                # inpaint_assign_ops = []
                # dis_assign_ops = []

                # for var in edge_gen_vars:
                #     vname = var.name
                #     from_name = vname
                #     var_value = tf.train.load_variable(os.path.join(edge_model_dir, 'model'), from_name)
                #     edge_assign_ops.append(tf.assign(var, var_value))
                # sess.run(edge_assign_ops)
                # print('Edge model loaded!')

                # for var in color_gen_vars:
                #     vname = var.name
                #     from_name = vname
                #     var_value = tf.train.load_variable(os.path.join(color_model_dir, 'model'), from_name)
                #     color_assign_ops.append(tf.assign(var, var_value))
                # sess.run(color_assign_ops)
                # print('Color model loaded!')

                # for var in inpaint_gen_vars:
                #     vname = var.name
                #     from_name = vname
                #     var_value = tf.train.load_variable(os.path.join(model_dir, 'model'), from_name)
                #     inpaint_assign_ops.append(tf.assign(var, var_value))
                # sess.run(inpaint_assign_ops)
                # print('Refine model loaded!')

                # for var in dis_vars:
                #     vname = var.name
                #     from_name = vname
                #     var_value = tf.train.load_variable(os.path.join(model_dir, 'model'), from_name)
                #     dis_assign_ops.append(tf.assign(var, var_value))
                # sess.run(dis_assign_ops)
                # print('Dis model loaded!')

                # add above

                saver.restore(sess, os.path.join(model_dir, 'model'))
                step = tf.train.load_variable(os.path.join(model_dir, 'model'), 'gen_global_step')
                epoch = step // num_batch - 1

            with open(os.path.join(log_dir, 'logs.csv'), 'a+') as f:
                mywrite = csv.writer(f)
                mywrite.writerow(['dis_loss', 'gen_loss', 'gen_gan_loss', 'gen_l1_loss', 'gen_content_loss',
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
                    print('-----------gen_content_loss: {}'.format(logs_[4]))
                    # print('-----------gen_style_loss: {}'.format(logs_[5]))

                    with open(os.path.join(log_dir, 'logs.csv'), 'a+') as f:
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
    model = RefineModel(cfg)
    model.train()
