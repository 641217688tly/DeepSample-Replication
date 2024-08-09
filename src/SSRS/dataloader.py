import os
import numpy as np

def load_mnist_images(file_path):
    import struct
    with open(file_path, 'rb') as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_mnist_labels(file_path):
    import struct
    with open(file_path, 'rb') as file:
        magic, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)
    return labels


def load_operational_dataset(data_dir='../../data/dataset/MNIST/raw'):
    # 拼接训练集, 测试集和对应标签文件的路径
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    # 加载MNIST官方划分的测试集和训练集
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    # 将MNIST官方划分的全部训练集和测试集中的前500个样本作为操作集
    operational_images = np.concatenate([train_images, test_images[:500]])  # 60500
    operational_labels = np.concatenate([train_labels, test_labels[:500]])  # 60500

    return operational_images, operational_labels
