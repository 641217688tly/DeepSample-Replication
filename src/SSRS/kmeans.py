import numpy as np

class Cluster: # Partition
    def __init__(self, id, centroid, dataset):
        self.id = id
        self.centroid = centroid # 质心
        self.dataset = dataset # 数据集
        self.avg_confidence = 0 # 平均置信度
        self.std = 0 # 标准差
        self.normalized_std = 0 # 归一化标准差
        self.sample_size = 0 # 样本数


# 1. 从待聚类的数据集中速记选择k个样本作为初始聚类中心
def init_centroids(dataset, k):
    """
    从待聚类的数据集中速记选择k个样本作为初始聚类中心
    :param dataset:
    :param k:
    :return:
    """
    # 如果k大于dataset的样本数, 则抛出异常
    if k > dataset.shape[0]:
        raise Exception("k is greater than the number of samples")
    # 从dataset中随机获取k个不重复的数据
    centroids = dataset[np.random.choice(dataset.shape[0], k, replace=False)]
    return centroids


# 2. 将dataset的每个样本分配到最近的聚类中心
def assign_cluster(dataset, centroids):
    """
    将dataset的每个样本分配到最近的聚类中心
    """
    # 计算每个点到所有质心的距离
    distances = np.sqrt(((dataset[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    # 找到最近的质心索引
    cluster_assignment = np.argmin(distances, axis=1)
    return cluster_assignment


# 3. 计算被分配到不同聚类中的样本到其所属聚类中心的距离并计算成本函数
def compute_cost(dataset, cluster_assignment, centroids):
    """
    计算被分配到不同聚类中的样本到其所属聚类中心的距离并计算成本函数
    """
    total_cost = 0
    for i in range(centroids.shape[0]):
        # 计算属于当前质心的所有点的距离平方和
        total_cost += np.sum((dataset[cluster_assignment == i] - centroids[i]) ** 2)
    return total_cost


# 4.初始化多组聚类的质心, 选择其中最优的一组
def find_best_centroids(dataset, k, num_iters=100):
    """
    初始化多组聚类的质心, 选择其中最优的一组
    """
    min_cost = np.inf
    best_centroids = None
    for _ in range(num_iters):
        centroids = init_centroids(dataset, k)
        cluster_assignment = assign_cluster(dataset, centroids)
        cost = compute_cost(dataset, cluster_assignment, centroids)
        if cost < min_cost:
            min_cost = cost
            best_centroids = centroids
    return best_centroids


# 5. 实现K-Means聚类算法
def kmeans(dataset, k, num_iters):
    """
    实现K-Means聚类算法
    """
    # 1. 初始化多组聚类的质心, 选择其中最优的一组
    centroids = find_best_centroids(dataset, k, num_iters)
    # 2. 重复执行以下步骤num_iters次
    for i in range(num_iters):
        # 3. 将dataset的每个样本分配到最近的聚类中心
        cluster_assignment = assign_cluster(dataset, centroids)
        # 4. 计算每个样本到其所属聚类中心的距离并计算成本函数, 然后打印结果
        cost = compute_cost(dataset, cluster_assignment, centroids)
        print(f"Iteration {i + 1}, Cost: {cost}")
        # 5. 更新聚类中各个质心的位置
        for j in range(k):
            centroids[j] = np.mean(dataset[cluster_assignment == j], axis=0)
    # 6.将聚类后的结果返回
    return centroids
