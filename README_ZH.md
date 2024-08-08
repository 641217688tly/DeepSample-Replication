# DeepSample

此仓库提供了DeepSample执行的相关文件。
通过运行脚本 'run_DeepSample.sh' 可以复现对9个用于分类的DNN和2个用于回归的DNN的实验。
一旦实验完成，所有结果将生成在 'Results' 文件夹中。

## DeepSample文件夹
"DeepSample" 文件夹包含所有的 ".jar" 文件和源代码。
源代码的组织如下：
- "main" 文件夹包含所有用于生成jar文件的java文件。
- "selector.classification" 包含为分类实现的采样算法的源代码，以及DeepEST。
- "selector.regression" 包含为回归实现的采样算法的源代码，以及DeepEST。
- "utility" 和 "utility.regression" 包含对选择器有用的工具和数据结构。

代码是在Eclipse中开发的。

## 其他采样器
CES和SRS的实现可在以下地址获得：'https://github.com/Lizenan1995/DNNOpAcc'。
复现结果可以通过：
1. 克隆仓库；
2. 在 'CE method/MNIST/normal' 中导入模型；
3. 在 'CE method/MNIST/normal' 中导入并运行我们仓库的 'CES_SRS' 文件夹中的文件（'crossentropy*.py' 文件）。

以下我们报告了运行CES所需的选定层（'crossentropy*.py' 中的LAY参数）：
- A: -2
- B: -4
- C: -2
- D: -4
- E: -4
- F: -2
- G: -2
- H: -4
- I: -4
- Dave_orig: -3
- Dave_dropout: -3

## 神经网络可用性
训练过的模型包含在 'models' 文件夹中。模型G、Dave_orig和Dave_dropout模型可在 'https://file.io/0405r74sgwqV' 下载。
'dataset' 文件夹包含了所有模型的数据集、预测结果和辅助变量。
分类模型的源代码可在 'https://github.com/ICOS-OAA/ICOS.git' 获得

## 论文结果
'Results_paper' 文件夹包含了论文中报告的结果。

为了方便实践者查询结果，例如根据主要目标（低RMSE、RMedSE或高故障暴露能力）和输入配置（例如，小样本大小、MNIST数据集、置信度作为辅助变量）查询表现最佳的技术（基于其在前三名排名中的出现次数），提供了一个Python笔记本（[notebook](./Results_paper/_Discussion/interactive_notebook/summary.ipynb)）

## 要求和依赖
提供的代码需要Java 8。
"libs" 文件夹包含运行实验所需的所有库。
