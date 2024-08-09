import os
import numpy as np
import tensorflow as tf


class Sample:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.confidence = None  # 置信度
        self.las = None
        self.dsa = None


class DataLoader:
    def __init__(self, model_path='../../data/model/modelA.h5', data_dir='../../data/dataset/MNIST/raw'):
        self.model_path = model_path
        self.data_dir = data_dir
        self.model = tf.keras.models.load_model(model_path)
        self.samples = self.load_operational_dataset()
        # 初始化置信度, DSA和LAS
        self.set_confidence()
        self.set_dsa()
        self.set_las()

    def load_mnist_images(self, file_path):
        import struct
        with open(file_path, 'rb') as file:
            magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
            images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
        return images

    def load_mnist_labels(self, file_path):
        import struct
        with open(file_path, 'rb') as file:
            magic, num = struct.unpack(">II", file.read(8))
            labels = np.fromfile(file, dtype=np.uint8)
        return labels

    def load_operational_dataset(self):
        # 拼接训练集, 测试集和对应标签文件的路径
        train_images_path = os.path.join(self.data_dir, 'train-images-idx3-ubyte')
        train_labels_path = os.path.join(self.data_dir, 'train-labels-idx1-ubyte')
        test_images_path = os.path.join(self.data_dir, 't10k-images-idx3-ubyte')
        test_labels_path = os.path.join(self.data_dir, 't10k-labels-idx1-ubyte')
        # 加载MNIST官方划分的测试集和训练集
        train_images = self.load_mnist_images(train_images_path)
        train_labels = self.load_mnist_labels(train_labels_path)
        test_images = self.load_mnist_images(test_images_path)
        test_labels = self.load_mnist_labels(test_labels_path)

        # 将MNIST官方划分的全部训练集和测试集中的前500个样本作为操作集
        operational_images = np.concatenate([train_images, test_images[:500]])  # 60500
        operational_labels = np.concatenate([train_labels, test_labels[:500]])  # 60500

        # 创建Sample类的列表
        samples = []
        for image_data, label_data in zip(operational_images, operational_labels):
            # 创建一个Sample对象并添加到列表中
            sample = Sample(data=image_data, label=label_data)
            samples.append(sample)
        return samples

    def set_confidence(self):
        print(self.model.summary())
        # 将Sample对象列表中的数据提取出来并转换为模型输入格式
        images = np.array([sample.data for sample in self.samples])
        # 归一化到0-1范围
        images /= 255.0
        # 使用模型进行预测
        predictions = self.model.predict(images)
        # 更新每个Sample对象的置信度
        confidences = np.max(predictions, axis=1)
        for sample, confidence in zip(self.samples, confidences):
            sample.confidence = confidence

    def set_las(self):
        pass

    def set_dsa(self):
        pass

