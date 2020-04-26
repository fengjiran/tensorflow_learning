import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from load_mnist import load_mnist_image
from load_mnist import load_mnist_label


def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def plot_embedding_3D(data, label, title):
    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure().add_subplot(111, projection='3d')
    for i in range(data.shape[0]):
        fig.text(data[i, 0], data[i, 1], data[i, 2], str(label[i]),
                 color=plt.cm.Set1(label[i]), fontdict={'weight': 'bold', 'size': 9})
    return fig


def main():
    data = load_mnist_image('/Users/richard/Desktop/mnist', 'train-images-idx3-ubyte')
    label = load_mnist_label('/Users/richard/Desktop/mnist', 'train-labels-idx1-ubyte')

    print('Begining......')
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    result_2D = tsne_2D.fit_transform(data)
    tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
    result_3D = tsne_3D.fit_transform(data)
    print('Finished......')

    fig1 = plot_embedding_2D(result_2D, label, 't-SNE')
    plt.show(fig1)
    fig2 = plot_embedding_3D(result_3D, label, 't-SNE')
    plt.show(fig2)


if __name__ == '__main__':
    main()
