import os
import tensorflow as tf


def parse_tfrecord(example_proto):
    features = {'shape': tf.FixedLenFeature([3], tf.int64),
                'data': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], tf.float32)
    img = tf.reshape(data, parsed_features['shape'])
    img = tf.image.resize_images(img, [315, 256])
    img = tf.random_crop(img, [256, 256, 3])

    return img


filenames = tf.placeholder(tf.string, shape=[None])
# dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parse_tfrecord)
# dataset = dataset.map(input_parse)
dataset = dataset.shuffle(buffer_size=5000)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(16))
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
batch_data = iterator.get_next()

compress_path = 'F:\\Datasets\\celeba_tfrecords'
for _, _, files in os.walk(compress_path):
    tfrecord_filename = files
tfrecord_filename = [os.path.join(compress_path, file) for file in tfrecord_filename]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(iterator.initializer, feed_dict={filenames: tfrecord_filename})

    for i in range(10):
        batch = sess.run(batch_data)
        print(batch.shape)
