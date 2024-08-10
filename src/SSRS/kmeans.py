import numpy as np


class Partition:
    """
    一个Partition对象代表一个分区, 包含了分区的质心, 位于该分区的所有样本, 以及在完成聚类后分区的一些统计信息
    """

    def __init__(self, uid, centroid=None, samples=None, auv_type="confidence"):
        self.uid = uid
        self.centroid = centroid  # 质心值或质心向量
        self.samples = samples  # 属于该分区的所有样本
        self.auv_type = auv_type  # "confidence" / "las" / "dsa"
        self.avg_auv = 0  # 求辅助变量的均值
        self.std = 0  # 标准差
        self.normalized_std = 0  # 归一化标准差
        self.n_p = 0  # 从该分区中采样的样本数


class KMeans:
    """
    一个KMeans对象代表一个KMeans聚类器, 负责对给定的数据集按照指定的辅助变量进行聚类
    """

    def __init__(self, dataset, k=10, num_iters=25, auv_type="confidence"):
        print("------------------------Initializing K-Means------------------------")
        self.dataset = dataset  # 所有待聚类的样本
        self.k = k  # 聚类的簇数, 默认为10
        self.auv_type = auv_type  # 按照当前辅助变量进行分区, "confidence" / "las" / "dsa"
        self.num_iters = num_iters  # k-means的迭代次数
        self.partitions = self.initialize_partitions()  # k个分区/簇

    def randomly_generate_centroids(self):
        """
        从待聚类的数据集中速记选择k个样本作为初始聚类中心
        """
        # 如果k大于dataset的样本数, 则抛出异常
        if self.k > len(self.dataset):
            raise Exception("k is greater than the number of samples")
        # 先获得k个不重复的索引
        indices = np.random.choice(len(self.dataset), self.k, replace=False)
        # 从样本中随机获取k个不重复的数据, 然后用这些样本构造Partition
        partitions = []
        for i, index in enumerate(indices):
            partition = Partition(uid=i, centroid=getattr(self.dataset[index], self.auv_type, None),
                                  auv_type=self.auv_type)
            partitions.append(partition)
        return partitions

    def assign_partition(self, partitions):
        """
        将dataset的每个样本分配到与质心距离最近的分层中
        """
        # 先清空每个分层的样本
        for partition in partitions:
            partition.samples = []
        # 然后遍历dataset
        for sample in self.dataset:
            min_distance = np.inf
            min_partition = None
            # 遍历每个分层
            for partition in partitions:
                # 计算质心和样本之间的平方差距离
                distance = np.sum((getattr(sample, self.auv_type, None) - partition.centroid) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    min_partition = partition
            # 将样本分配到距离最近的分层中
            sample.partition = min_partition  # 更新样本的分区
            min_partition.samples.append(sample)

    def compute_cost(self, partitions):
        """
        计算被分配到不同聚类中的样本到其所属聚类中心的距离并计算成本函数
        """
        total_cost = 0
        for partition in partitions:  # 遍历每个分层
            cost = 0
            for sample in partition.samples:  # 遍历分层中的每个样本
                cost += np.sum((getattr(sample, self.auv_type, None) - partition.centroid) ** 2)
            total_cost += cost
        return total_cost

    def initialize_partitions(self):
        """
        用于初始化最优分层, 该函数将生成多组分层, 选择其中初始质心最优的一组
        """
        min_cost = np.inf
        best_partitions = None
        for i in range(10):
            partitions = self.randomly_generate_centroids()
            self.assign_partition(partitions)
            cost = self.compute_cost(partitions)
            if cost < min_cost:
                min_cost = cost
                best_partitions = partitions
                print(f"Initializing greater partitions' centroid, current cost: {cost}")
        return best_partitions

    def cluster(self):
        """
        开始聚类
        """
        for i in range(self.num_iters):
            # 1. 将dataset的每个样本分配到与质心距离最近的分层中
            self.assign_partition(self.partitions)
            # 2. 更新每个分层的质心
            for partition in self.partitions:
                if len(partition.samples) == 0:
                    continue
                # 获取partition.samples中所有样本的辅助变量的均值
                partition.centroid = np.mean([getattr(sample, self.auv_type, None) for sample in partition.samples])

            if i % 2 == 0:  # 每隔2次迭代输出一次信息
                print(f"K-Means Iteration {i}, current cost: {self.compute_cost(self.partitions)}")
            # 如果是最后一次迭代, 则输出最终的cost
            if i == self.num_iters - 1:
                # 将所有样本的partition属性进行更新(因为最后一次迭代时每个partition质心的值会有微量的更新)
                self.assign_partition(self.partitions)

        print("------------------------K-Means Cluster Done!------------------------\n")
        return self.partitions
