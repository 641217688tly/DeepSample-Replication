from src.SSRS.dataloader import DataLoader
from src.SSRS.estimator import Estimator
from src.SSRS.kmeans import KMeans
from src.SSRS.sampling import Sampler

if __name__ == '__main__':
    csv_path = '../../data/dataset/MNIST/modelA_final.csv'
    dataloader = DataLoader(csv_path=csv_path)  # 加载数据
    samples = dataloader.samples

    kmeans = KMeans(samples, k=10, num_iters=50, auv_type="lsa")  # 初始化KMeans
    partitions = kmeans.cluster()  # 获得聚类后的分层

    sampler = Sampler(partitions, budget=800)  # 初始化采样器
    test_set = sampler.sample()  # 采样

    estimator = Estimator(test_set, csv_path=csv_path)  # 初始化评估器
    estimator.estimate()  # 复现
