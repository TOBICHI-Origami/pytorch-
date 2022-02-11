# pytorch-情感分类
### 1.作用:利用pytorch对电影影评进行情感分类(2分类问题)
### 2.流程

  * 构建单词表(库内置IDMB电影影评数据 利用 dataset 和 dataloader 构建单词表 对每个 comment 分词并构建索引表 调整每个comment的大小使之一致)
  * 随机初始化64维度的向量 
  * 训练修改参数(利用GNN门卷积网络, 利用Googlecolaboratory训练) 
  * 保存模型(torch.save) 
  * 测试并算出正确率以及损失函数(利用交叉墒)
