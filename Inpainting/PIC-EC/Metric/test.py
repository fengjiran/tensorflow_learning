import os
from glob import glob
from frechet_kernel_Inception_distance import *
from inception_score import *

filenames = glob(os.path.join('./fake', '*.*'))
images = [get_images(filename) for filename in filenames]
images = np.transpose(images, axes=[0, 3, 1, 2])

# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 1
batch_size = 1

# Run images through Inception.
inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
logits = inception_logits(inception_images)
print(logits)
assert(type(images) == np.ndarray)
assert(len(images.shape) == 4)
assert(images.shape[1] == 3)
assert(np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'

n_batches = len(images) // batch_size
print(n_batches)

preds = np.zeros([n_batches * batch_size, 1000], dtype=np.float32)
for i in range(n_batches):
    inp = images[i * batch_size:(i + 1) * batch_size] / 255. * 2 - 1
    preds[i * batch_size:(i + 1) * batch_size] = logits.eval({inception_images: inp})[:, :1000]
preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)

print(preds)
print(preds.shape[0])

# kl = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
# kl = np.mean(np.sum(kl, 1))
# score = np.exp(kl)
# print(score)

scores = []
splits = 10
for i in range(splits):
    part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
    print(part)
    kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    scores.append(np.exp(kl))

print(scores)
# IS = get_inception_score(BATCH_SIZE, images, inception_images, logits, splits=10)

print('done')
