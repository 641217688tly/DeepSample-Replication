import numpy as np
import tensorflow as tf

class Estimator:
    def __init__(self, test_set, model_path='../../data/model/modelA.h5'):
        print("------------------------Initializing Estimator------------------------")
        self.test_set = test_set
        self.model = tf.keras.models.load_model(model_path)

    def estimate(self):
        successes = []
        failures = []
        for sample in self.test_set:
            data = sample.data
            label = sample.label
            data = data.reshape(1, 28, 28, 1)
            prediction = self.model.predict(data)
            predicted_label = np.argmax(prediction, axis=1)[0]  # 获取预测的类别索引
            if predicted_label == label:
                successes.append(sample)
            else:
                failures.append(sample)
        accuracy = len(successes) / len(self.test_set)
        print(f"Model Accuracy: {accuracy}")
        print(f"Number of successful predictions: {len(successes)}")
        print(f"Number of failed predictions: {len(failures)}")
        print("------------------------Estimating Done!------------------------\n")