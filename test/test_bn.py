import tensorflow as tf

x = tf.placeholder(tf.float32, [128, 112, 112, 3])

bn1 = tf.keras.layers.BatchNormalization()


y1 = bn1(x, training=True)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
update_ops.extend(bn1.updates)
# train_ops = tf.group([minimization_op, update_ops])

print(update_ops)
print(len(update_ops))
