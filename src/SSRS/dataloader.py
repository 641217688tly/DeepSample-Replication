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
        self.partition = None
        self.predicted_label = None


class DataLoader:
    def __init__(self, model_path='../../data/model/modelA.h5', data_dir='../../data/dataset/MNIST/raw'):
        print("------------------------Loading Data------------------------")
        self.model_path = model_path
        self.data_dir = data_dir
        self.model = tf.keras.models.load_model(model_path)
        self.samples = self.load_operational_dataset()
        # 初始化置信度, DSA和LAS
        self.load_confidence()
        self.load_dsa()
        self.load_las()
        print("------------------------Data loaded successfully!------------------------\n")

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

    def load_confidence(self):
        print("Loading confidence...")
        # 确保 TensorFlow 能使用 GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # 设置 TensorFlow 使其在每个 GPU 上尽可能多地使用内存
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print("Error setting GPU: ", e)

        # 将Sample对象列表中的数据提取出来并转换为模型输入格式
        images = np.array([sample.data for sample in self.samples], dtype=np.float32) / 255.0

        # 使用模型进行预测后更新每个Sample对象的置信度
        predictions = self.model.predict(images)
        confidences = np.max(predictions, axis=1)
        for sample, confidence in zip(self.samples, confidences):
            sample.confidence = 1 - confidence  # 原文假设辅助变量与准确性负相关(即辅助变量$𝜒_i$的值越大, DNN模型对样本$d_i$预测的置信度越低, 那么DNN越有可能产生误判), 因此$𝜒_i = 1- C_{d_i}$

    def load_las(self):
        pass

    def load_dsa(self):
        pass
