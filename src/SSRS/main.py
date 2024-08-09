from kmeans import *
from dataloader import *

if __name__ == '__main__':
    model_path = '../../data/model/modelA.h5'
    data_dir = '../../data/dataset/MNIST/raw'
    dataloader = DataLoader(model_path=model_path, data_dir=data_dir)  # 加载数据
    kmeans = KMeans(dataloader.samples, k=10, num_iters=100, auv_type="confidence")  # 初始化KMeans
    partitions = kmeans.cluster()  # 获得聚类后的分层

