import numpy as np
import tensorflow as tf
import csv


class Estimator:
    def __init__(self, test_set, model_path=None, csv_path=None):
        print("------------------------Initializing Estimator------------------------")
        self.test_set = test_set  # 这是一个字典, partition对象为键,采样得到的样本数组为值
        self.csv_path = csv_path
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path) if self.model_path is not None else None
        self.calculate_theta_p()
        # self.debug()

    def debug(self):
        for sample in self.test_set:
            print(f'Partition{sample.partition.uid}(Centroid Value:{sample.partition.centroid})')

    def calculate_theta_p(self):
        # self.test_set是一个字典, partition对象为键,采样得到的样本数组为值
        for partition, samples in self.test_set.items():
            failures = 0
            successes = 0
            if self.model_path:
                # 使用模型预测每个分区下所有样本的标签
                images = np.array([sample.data for sample in samples], dtype=np.float32)
                predictions = self.model.predict(images)
                predicted_labels = np.argmax(predictions, axis=1)  # 得到每一行中最大值的索引
                for i, (sample, predicted_label) in enumerate(zip(samples, predicted_labels)):
                    sample.predicted_label = predicted_label
                    if predicted_label != sample.label:
                        failures += 1
                    else:
                        successes += 1
            elif self.csv_path:
                failures = 0
                successes = 0
                for i, sample in enumerate(samples):
                    if sample.predicted_label != sample.label:
                        failures += 1
                    else:
                        successes += 1
            partition.failures = failures
            partition.successes = successes
            partition.theta_p = round(failures / (partition.n_p + 1e-6), 5)
            print(f"Partition{partition.uid} theta_p: {partition.theta_p}")

    def estimate(self):
        results = []
        theta_p = 0
        successes = 0
        failures = 0
        for partition, samples in self.test_set.items():
            theta_p = theta_p + partition.theta_p * partition.n_p
            successes += partition.successes
            failures += partition.failures
            for sample in samples:
                results.append([len(results) + 1,
                                'pass' if sample.predicted_label == sample.label else 'fail',
                                sample.label,
                                sample.predicted_label,
                                f'Partition{partition.uid}(Centroid Value:{partition.centroid})',
                                sample.confidence if sample.confidence is not None else 'null',
                                sample.dsa if sample.dsa is not None else 'null',
                                sample.lsa if sample.lsa is not None else 'null'])
        theta_p = round(theta_p / len(results), 5)  # accuracy = 1 - theta_p

        # 输出结果到csv文件
        csv_file_name = ''
        if self.model_path:
            csv_file_name = self.model_path.split('/')[-1].split('.')[0]
        elif self.csv_path:
            csv_file_name = 'replicate_' + self.csv_path.split('/')[-1].split('.')[0]
        with open(f'../../data/results/{csv_file_name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'Successes: {successes}',
                             f'Failures: {failures}',
                             f'Budget: {len(self.test_set)}',
                             f'Model Accuracy: {1 - theta_p:.4f}'])
            # 表头: ID(自增),outcome(模型预测的结果是正确还是错误),label(数据的标签),partition(数据的所在分区),confidence(置信度),dsa(如果没有则为None),lsa(如果没有则为None)
            writer.writerow(
                ['ID', 'outcome', 'label', 'predicted label', 'partition', 'confidence(= 1 - Inference Probability)',
                 'dsa', 'lsa'])
            writer.writerows(results)

        print(f"Model Accuracy: {1 - theta_p}")
        print(f"Number of successful predictions: {successes}")
        print(f"Number of failed predictions: {failures}")
        print("------------------------Estimating Done!------------------------\n")
