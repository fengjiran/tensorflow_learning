import tensorflow as tf
import tensorflow_probability as tfp

alpha = 1
k = 2
bs = 10
num_classes = 6
label = tf.random.uniform([bs, k], maxval=num_classes, dtype=tf.int32)

dist = tfp.distributions.Dirichlet([alpha] * k)
lam_aug = dist.sample([bs])
lam_aug = tf.reshape(lam_aug, [bs, k, 1])
label = tf.one_hot(label, depth=num_classes)
soft_targets = tf.reduce_sum(label * lam_aug, axis=1)

print(soft_targets)
with tf.Session() as sess:
    res = sess.run(soft_targets)
    print(res)
