# 预备知识
D2L在线学习的第二张内容

##数据操作（ndarray）2022-11-28
1. import torch基本运算

2. 广播机制
    两个张量的维度大小向右对齐，触发广播机制的条件：
    a. 同一维度大小相等；
    或 b. 某个维度 一个张量有，另一个张量没有 ；
    或 c. 某个维度 一个张量有，另一个张量也有且大小不同，但大小是1。
3. 深度学习存储和操作数据的主要接口是张量，torch.tensor(x,x)

##数据预处理
1. pandas软件包是Python中常用的数据分析工具中，pandas可以与张量兼容。
    import torch
    X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)

    这里inputs和outputs是产生于data
    import pandas as pd
    data = pd.read_csv(data_file)

2. 用pandas处理缺失的数据时，我们可根据情况选择用插值法和删除法。
    插值法：
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean())
    删除法（变成新的一列）：
    inputs = pd.get_dummies(inputs, dummy_na=True)