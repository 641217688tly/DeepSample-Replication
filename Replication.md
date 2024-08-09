# Setup

**Technique**

分层简单随机抽样(Stratified Simple Random Sampling, SSRS)

**Auxiliary Variable**

选择置信度(Confidence)作为辅助变量$𝜒$

**Dataset**

MNIST数据集

+ 训练集: 7,000
+ 验证集: 2,500
+ 操作数据集: 60,500(规模较大)



# Methodology

## 1. Collect Confidence

从MNIST中分割出操作数据集, 然后将这些操作数据集中的数据作为DNN的输入以获取操作数据集中所有样本的置信度

## 2. Apply K-Means

使用K-Means按照置信度对操作数据集进行聚类, 其中$k =10$

## 3. Handle Clusters

对于得到的十个分区, 先计算每个分区的平均置信度$\bar{x}_p$:
$$
\bar{x}_p = \frac{1}{N_p} \sum_{i=1}^{N_p}x_i
$$
然后逐个计算每个分区的标准差$\sigma_p$:
$$
\sigma_p = \sqrt{\frac{1}{N_p - 1} \sum_{i=1}^{N_p} (x_i - \bar{x}_p)^2}
$$


然后归一化该标准差, 使得这10个分区的标准差之和为1, 即($\sigma_{p_1} + \sigma_{p_2}+ ... + \sigma_{p_{10}} =1$):
$$
\text{Normalized } \sigma_p = \frac{\sigma_p}{\sum_{p=1}^{10} \sigma_p}
$$

## 4. Sampling

利用先前处理分区得到的信息, 计算每个分区应该抽取的样本数:
$$
n_p = N \times \frac{\sigma_p \cdot N_p}{\sum_{p=1}^P (\sigma_p \cdot N_p)}
$$
之后, 根据求得的每个分区需要提供的样本数进行无放回的随机抽样, 用抽得的样本构建测试数据集$T$



## 5.Evaluate DNN

使用抽样得到的测试数据集$T$来评估DNN的准确率, 同时统计被DNN误分类的样本的数量



## 6. Repetition

在模型modelA, modelB和modelC上分别重复上述实验步骤, 预计每个模型重复30次实验



# Result

![img.png](E:/Tools/Windows/Typora/Typora%20Image%20Cache/Replication/img.png)

