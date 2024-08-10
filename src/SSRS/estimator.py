import numpy as np
import tensorflow as tf
import csv


class Estimator:
    def __init__(self, test_set, model_path='../../data/model/modelA.h5'):
        print("------------------------Initializing Estimator------------------------")
        self.test_set = test_set
        self.model_name = model_path.split('/')[-1].split('.')[0]
        self.model = tf.keras.models.load_model(model_path)
        #self.debug()

    def debug(self):
        for sample in self.test_set:
            print(f'Partition{sample.partition.uid}(Centroid Value:{sample.partition.centroid})')

    def estimate(self):
        # 转换数据为模型输入格式并归一化
        images = np.array([sample.data for sample in self.test_set], dtype=np.float32)

        # 使用模型进行预测
        predictions = self.model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)  # 得到每一行中最大值的索引

        # 更新每个Sample对象的预测标签, 并统计成功和失败的预测
        successes = 0
        results = []
        for i, (sample, predicted_label) in enumerate(zip(self.test_set, predicted_labels)):
            sample.predicted_label = predicted_label
            if predicted_label == sample.label:
                successes += 1
            else:
                sample.predicted_label = predicted_label
            results.append([i + 1,
                            'pass' if predicted_label == sample.label else 'failed',
                            sample.label,
                            predicted_label,
                            f'Partition{sample.partition.uid}(Centroid Value:{sample.partition.centroid})',
                            sample.confidence, sample.dsa if sample.dsa != None else 'null',
                            sample.las if sample.las != None else 'null'])

        with open(f'../../data/results/{self.model_name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # 表头: ID(自增),outcome(模型预测的结果是正确还是错误),label(数据的标签),partition(数据的所在分区),confidence(置信度),dsa(如果没有则为None),lsa(如果没有则为None)
            writer.writerow(['ID', 'outcome', 'label', 'predicted label', 'partition', 'confidence', 'dsa', 'lsa'])
            writer.writerows(results)

        # 计算准确率
        accuracy = successes / len(self.test_set)
        print(f"Model Accuracy: {accuracy}")
        print(f"Number of successful predictions: {successes}")
        print(f"Number of failed predictions: {len(self.test_set) - successes}")
        print("------------------------Estimating Done!------------------------\n")
