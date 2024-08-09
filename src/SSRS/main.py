from kmeans import *
from dataloader import *
from sampling import *

if __name__ == '__main__':
    model_path = '../../data/model/modelA.h5'
    data_dir = '../../data/dataset/MNIST/raw'
    dataloader = DataLoader(model_path=model_path, data_dir=data_dir)  # 加载数据
    kmeans = KMeans(dataloader.samples, k=10, num_iters=20, auv_type="confidence")  # 初始化KMeans
    partitions = kmeans.cluster()  # 获得聚类后的分层
    sampler = Sampler(partitions, budget=800)  # 初始化采样器
    test_set = sampler.sample()  # 采样



