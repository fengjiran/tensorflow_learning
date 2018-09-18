import tensorflow as tf

logits = tf.constant([[1.0, 2.0, 3.0],
                      [1.0, 2.0, 3.0],
                      [1.0, 2.0, 3.0]])
y = tf.nn.softmax(logits)
labels = tf.constant([[0, 0, 1.0],
                      [0, 0, 1.0],
                      [0, 0, 1.0]])  # sparse labels
dense_labels = tf.argmax(labels, 1)

tf_log = tf.log(y)
mult = tf.multiply(labels, tf_log)
ce1 = -tf.reduce_sum(mult)
# ce1 = -tf.reduce_mean(tf.reduce_sum(mult, 1))

ce2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
ce2 = tf.reduce_sum(ce2)

ce3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=dense_labels)
ce3 = tf.reduce_sum(ce3)

with tf.Session() as sess:
    ce_value1, ce_value2, ce_value3 = sess.run([ce1, ce2, ce3])
    print(ce_value1, ce_value2, ce_value3)
