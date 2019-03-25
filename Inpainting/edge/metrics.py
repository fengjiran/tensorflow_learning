import tensorflow as tf

# use tf.map_fn


def edge_accuracy(real_edge, fake_edge, threshold):
    lables = real_edge > threshold
    fake_edge = fake_edge > threshold

    relevant = tf.reduce_sum(tf.cast(lables, dtype=tf.float32), axis=(1, 2, 3))
    selected = tf.reduce_sum(tf.cast(fake_edge, dtype=tf.float32), axis=(1, 2, 3))

    precision = []
    recall = []
    # if relevant == 0 and selected == 0:
    #     precision = 1
    #     recall = 1
    # else:
    #     true_positive = tf.cast((lables == fake_edge) * lables, dtype=tf.float32)
    #     recall = tf.reduce_sum(true_positive) / (relevant + 1e-8)
    #     precision = tf.reduce_sum(true_positive) / (selected + 1e-8)

    return precision, recall
