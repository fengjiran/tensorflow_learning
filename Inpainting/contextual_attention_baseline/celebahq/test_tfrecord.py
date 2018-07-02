import cv2
import numpy as np
import tensorflow as tf


def parse_tfrecord(example_proto):
    features = {'shape': tf.FixedLenFeature([3], tf.int64),
                'data': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], tf.uint8)
    # img = tf.reshape(data, parsed_features['shape'])
    img = tf.reshape(data, [1024, 1024, 3])

    return img


val_tfrecord_filenames = "F:\\Datasets\\celebahq_tfrecords\\val\\celebahq_valset.tfrecord-001"
val_filenames = tf.placeholder(tf.string)
val_data = tf.data.TFRecordDataset(val_filenames)
val_data = val_data.map(parse_tfrecord)
val_data = val_data.batch(1)
# val_data = val_data.repeat()
val_iterator = val_data.make_initializable_iterator()
val_batch_data = val_iterator.get_next()
val_batch_data = tf.image.resize_area(val_batch_data, [256, 256])
# val_batch_data = tf.clip_by_value(val_batch_data, 0., 255.)
val_batch_data = val_batch_data / 127.5 - 1

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    sess.run(val_iterator.initializer, feed_dict={val_filenames: val_tfrecord_filenames})
    val_batch_data = sess.run(val_batch_data)
    print(val_batch_data.shape)
    val_batch_data = np.reshape(val_batch_data, (256, 256, 3))
    val_batch_data = (val_batch_data + 1) * 127.5
    val_batch_data = val_batch_data.astype(np.uint8)
    cv2.imwrite('F:\\test.png', val_batch_data[:, :, ::-1])
