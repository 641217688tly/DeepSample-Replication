import numpy as np


class Sampler:
    def __init__(self, partitions, budget=800):
        print("------------------------Initializing Sampler------------------------")
        self.partitions = partitions
        self.budget = budget
        self.test_set = []
        self.calculate_stats()

    def calculate_stats(self):
        """计算每个分区的统计信息并进行采样准备。"""
        self.calculate_avg_auv()
        self.calculate_std()
        self.calculate_normalized_std()
        self.calculate_n_p()

    def calculate_avg_auv(self):
        """计算每个分区的平均辅助变量值。"""
        for partition in self.partitions:
            aux_values = [getattr(sample, partition.auv_type) for sample in partition.samples]
            partition.avg_auv = np.mean(aux_values)

    def calculate_std(self):
        """计算每个分区的标准差。"""
        for partition in self.partitions:
            aux_values = [getattr(sample, partition.auv_type) for sample in partition.samples]
            partition.std = np.std(aux_values, ddof=1)

    def calculate_normalized_std(self):
        """计算每个分区的归一化标准差。"""
        std_values = [partition.std for partition in self.partitions]
        total_std = sum(std_values)
        for partition in self.partitions:
            partition.normalized_std = round(partition.std / total_std, 5)

    def calculate_n_p(self):
        """计算每个分区的采样样本数。"""
        weighted_stds = [partition.normalized_std * len(partition.samples) for partition in self.partitions]
        total_weighted_std = sum(weighted_stds)
        for partition in self.partitions:
            partition.n_p = round(
                self.budget * (partition.normalized_std * len(partition.samples) / total_weighted_std)) # 四舍五入

    def sample(self):
        """从每个分区中采样，并将采样结果存入test_set后返回。"""
        for partition in self.partitions:
            sampled_samples = np.random.choice(partition.samples, partition.n_p, replace=False)
            self.test_set.extend(sampled_samples)
            print(f"Sampled {partition.n_p} samples from partition {partition.uid}")
        print(f"Total number of samples: {len(self.test_set)}")
        print("------------------------Sampling Done!------------------------\n")
        return self.test_set
