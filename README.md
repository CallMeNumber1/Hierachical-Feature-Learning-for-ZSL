#### 目标
- 构建层次化的特征表示，将不同级别的特征进行分层认知，层次化的标签对应不同级别的wordnet中的上下位词。
- - 期望利用较高层次的通用特征和较低层次的判别性特征。
- 将不同级别的特征进行分层认知，层次化的标签对应不同级别的上下位词
- - 先认知整体，再认知部分，通过多层次特征的拼接进行认知。
> 附加问题：提出的不同层次的特征如何量化迁移性的好坏？
#### 在AwA2数据集上进行实验
1. 为了进行层次化的特征学习，需要事先确定好动物类别在wordnet中的层次，可视化如下。首先使用属性embedding（85维）作为语义信息来监督训练。
2. 得到的9个上位类别为：[‘aquatic_mammal’, ‘bat’, ‘carnivore’, ‘insectivore’, ‘lagomorph’, ‘primate’, ‘proboscidean’, ‘rodent’, ‘ungulate‘]，目前实验中上位类别的属性embedding为其下位动物的平均值。
![image](https://github.com/CallMeNumber1/Hierachical-Feature-Learning-for-ZSL/blob/master/img/awa2%20in%20wordnet.png)
#### 网络架构
- 以resnet作为backbone，尝试了使用两层和三层进行实验。以两层为例做图：
![image](https://github.com/CallMeNumber1/Hierachical-Feature-Learning-for-ZSL/blob/master/img/network.png)
#### 两层时，GZSL下的实验结果
根据不同的实现方式分别做了实验，结果如下。
1. 双分支，各分支独立预测，两个分支损失相加来反向传播优化网络：
- - 双分支效果：最高34.66
- - 单分支效果：最高34.09
![image](https://github.com/CallMeNumber1/Hierachical-Feature-Learning-for-ZSL/blob/master/img/exp1-1.png)
![image](https://github.com/CallMeNumber1/Hierachical-Feature-Learning-for-ZSL/blob/master/img/exp1-2.png)
2. 双分支，各分支独立预测，将两个分支损失相加来反向传播优化网络。提前预训练了上位分支。
- - 双分支效果：最高32.27，最终在20~23间波动，效果反而不如没有预训练的好。
- - 单分支效果：最高34.09，最终在28.3~31.3间波动。
![image](https://github.com/CallMeNumber1/Hierachical-Feature-Learning-for-ZSL/blob/master/img/exp2-1.png)
![image](https://github.com/CallMeNumber1/Hierachical-Feature-Learning-for-ZSL/blob/master/img/exp2-2.png)
3. 特征拼接，最后将上位分支和主分支的结果合并（相加）用于预测。
- - 效果：最高31.98 最终在27.5~29.6间波动。
- - 虽然相对单分支效果没有变好，但更稳定了一点。
![image](https://github.com/CallMeNumber1/Hierachical-Feature-Learning-for-ZSL/blob/master/img/exp3-1.png)
![image](https://github.com/CallMeNumber1/Hierachical-Feature-Learning-for-ZSL/blob/master/img/exp3-2.png)
> 之前也看到过同类文章，在同一个卷积神经网络的不同层次学习特征，最后进行层次特征的融合。这样层次之间也相当于有了交互。
直接相加的原因是因为最后一层是权重固定为动物embedding的。因此相加才能保证最终的维度为[bs, embedding_size]，如果concat的话，维度就变化了。
#### 结果分析
- 目前多分支效果没有那么显著的原因可能是属性embedding上的层次相关的信息不够充足，下一把计划将使用的属性embedding替换为wordnet上训练的word embedding，进一步查看利用层次信息的效果。
- 实际上是要替换成普通的word embedding和wordnet上的word embedding相结合的embedding，前者提供语义信息，后者提供层次（结构）信息。