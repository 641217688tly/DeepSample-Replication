import sys

from src.SSRS.dataloader import DataLoader
from src.SSRS.estimator import Estimator
from src.SSRS.kmeans import KMeans
from src.SSRS.sampling import Sampler

if __name__ == '__main__':
    csv_dir = '../../data/dataset/MNIST/modelA_final.csv'
    dataloader = DataLoader(result_path=csv_dir)  # 加载数据
    samples = dataloader.samples

    kmeans = KMeans(samples, k=10, num_iters=50, auv_type="confidence")  # 初始化KMeans
    partitions = kmeans.cluster()  # 获得聚类后的分层

    sampler = Sampler(partitions, budget=800)  # 初始化采样器
    test_set = sampler.sample()  # 采样

    estimator = Estimator(test_set)  # 初始化评估器
    estimator.replicate()  # 复现
