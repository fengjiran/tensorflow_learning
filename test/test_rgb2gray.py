import tensorflow as tf
img_file = tf.read_file('F:\\Datasets\\celebahq\\img00000000.png')
img_decoded = tf.image.decode_png(img_file, channels=3)
img_gray = tf.image.rgb_to_grayscale(img_decoded)

with tf.Session() as sess:
    a, b = sess.run([img_decoded, img_gray])
    print(a.shape, b.shape)
    print(a[:, :, 0])
    print(b[:, :, 0])
