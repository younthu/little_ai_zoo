# MLP for Name to Gender in pytorch

`name2gender.py` 实现了一个简单的MLP模型，用于预测姓名的性别。
1. 3层FC + ReLU, 最后加一个SoftMax做分类。
1. 输入层：3个特征，分别是姓名的长度、姓名的第一个字母的one-hot编码、姓名的最后一个字母的one-hot编码。
1. 输出层：2个类，分别是男和女。
1. 损失函数：交叉熵损失函数。
1. 优化器：SGD。
1. 训练集：15402452个姓名，每个姓名的性别已知。
1. 测试集：3850614个姓名，每个姓名的性别未知。

stats:

1. Loss stuck at 0.6931
1. After refactor the model, and increase the epochs to 200,  the loss is stuck at 0.659
1. The default epoch takes 160s, with batch size 32, on M1 Pro. after batch size to 32 * 100, each epoch takes 40s. Per verification, the fast batch size is 1024 - 4096.
1. 