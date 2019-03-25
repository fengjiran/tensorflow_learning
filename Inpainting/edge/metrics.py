import tensorflow as tf


def edge_accuracy_per_image(real_edge, fake_edge, threshold):
    real_edge = real_edge > threshold
    fake_edge = fake_edge > threshold

    relevant = tf.reduce_sum(tf.cast(real_edge, dtype=tf.float32))
    selected = tf.reduce_sum(tf.cast(fake_edge, dtype=tf.float32))

    precision = 0
    recall = 0

    if relevant == 0 and selected == 0:
        precision = 1
        recall = 1
    else:
        true_positive = tf.cast((real_edge == fake_edge), dtype=tf.float32) * tf.cast(real_edge, dtype=tf.float32)
        recall = tf.reduce_sum(true_positive) / (relevant + 1e-8)
        precision = tf.reduce_sum(true_positive) / (selected + 1e-8)

    return precision, recall


def edge_accuracy(batch_real_edge, batch_fake_edge, threshold):
    elems = (batch_real_edge, batch_fake_edge)

    acc = tf.map_fn(fn=lambda x: edge_accuracy_per_image(x[0], x[1], threshold),
                    elems=elems,
                    dtype=(tf.float32, tf.float32))

    # precision = acc[:, 0]
    # recall = acc[:, 1]

    return acc  # tf.reduce_mean(precision), tf.reduce_mean(recall)


if __name__ == '__main__':
    a = tf.random_uniform([10, 256, 256, 1])
    b = tf.random_uniform([10, 256, 256, 1])
    acc = edge_accuracy_per_image(a, b, 0.5)
    print(acc[0].get_shape().as_list())
    print(acc)
    with tf.Session() as sess:
        print(sess.run(acc))
