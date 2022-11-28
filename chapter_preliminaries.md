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
    ```python {cmd=true}
    import torch
    X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
    ```
    这里inputs和outputs是产生于data
    ```python {cmd=true}
    import pandas as pd
    data = pd.read_csv(data_file)
    ```
2. 用pandas处理缺失的数据时，我们可根据情况选择用插值法和删除法。
    插值法：
    ```python {cmd=true}
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean())
    ```
    删除法（变成新的一列）：
    ```python {cmd=true}
    inputs = pd.get_dummies(inputs, dummy_na=True)
    ```
3. 作业1 删除NaN最多的那一列
    ```python
    import pandas as pd
    data = pd.read_csv(data_file)
    print(data)

    count_max = 0
    nan_numer = data.isnull().sum()
    print(nan_numer)
    id = nan_numer.idxmax()
    print(id)
    data_new = data.drop(id, axis=1)
    print(data_new)

    labels = ['NumRooms','Alley','Price']
    for label in labels:
        count = data[label].isna().sum()
        if count > count_max:
            count_max = count
            label_max = label
    data_new = data.drop(label_max, axis=1)
    print(data_new)
    ```
## 线性代数
1. 两个矩阵的按元素乘法称为Hadamard积（Hadamard product）（数学符号 ⊙ ）

2. 沿着行加和，相当于求每一列的和
   沿着列求和，相当于求每一行的和。
   ```python
    A_sum_axis0 = A.sum(axis=0) # 列和
    A_sum_axis1 = A.sum(axis=1) # 行和
   ```
3. 矩阵-向量积
    torch.mv(A, x)
    注意，A的列维数（沿轴1的长度）必须与x的维数（其长度）相同。
   矩阵-矩阵积
    torch.mm(A,B)
4. 在线性代数中，向量范数是将向量映射到标量的函数 𝑓
5. A/A.sum(axis=1) 会报错，因为广播机制而报错。所以通常要加上keepdims=True
    ```python
    print(A)
    sum_A = A.sum(axis=1, keepdims=True)
    print(sum_A)
    sum_B = A.sum(axis=1)
    print(sum_B)
    ```
    输出
    ```
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.],
            [16., 17., 18., 19.]])
    tensor([[ 6.],
            [22.],
            [38.],
            [54.],
            [70.]])
    tensor([ 6., 22., 38., 54., 70.])
    ```
