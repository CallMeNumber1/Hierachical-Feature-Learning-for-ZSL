#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :plot.py
@Date    :2021/01/10 11:54:01
@Author  :Chong
用来可视化单分支情况下的训练准确率和损失和测试准确率，以和多分支的情况进行对比
'''


import numpy as np
import matplotlib.pyplot as plt


def load(opt):
    '''
        加载npy格式的损失和准确率信息
    '''
    path = None
    if opt == 'adam':
        # path = '../'
        pass
    elif opt == 'sgd_monmentum_weight_decay':
        # path = './'
        pass
    elif opt == 'sgd_0.001':
        path = './sgd_0.001/'
    elif opt == 'sgd_steplr':
        path = './sgd_steplr/'
    elif opt == 'sgd_0.001_200epoch':
        path = './log/'
    train_acc = np.load(path+'train_acc_list.npy',allow_pickle=True)
    train_loss = np.load(path+'train_loss_list.npy',allow_pickle=True)
    test_acc = np.load(path+'test_acc_list.npy',allow_pickle=True)
    return train_acc, train_loss,test_acc
def annotate_max(acc):
    '''
    该函数用以在图上标注文字
    plt.annotate(s='str',
        xy=(x,y) ,
        xytext=(l1,l2) ,
        ...
    )
    s：为注释文本内容
    xy：为被注释的坐标点
    xytext：为注释文字的坐标位置
    '''
    acc_max = np.argmax(acc)
    # 只保留前几位数字，不然数字很长
    show_max = str(acc[acc_max])[0:5]
    plt.plot(acc_max, acc[acc_max], 'ko')
    plt.annotate(show_max, xy=(acc_max, acc[acc_max]), xytext=(acc_max, acc[acc_max]))
import argparse
if __name__ == '__main__':
    # 加载由哪个优化器训练出的模型
    opt = 'sgd_0.001_200epoch'
    train_acc, train_loss,test_acc = load(opt)

    epoch = [i for i in range(len(train_acc))]
    print(epoch)
    plt.figure(figsize=(20,10), dpi=80)
    plt.subplot(121)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    # plt.yticks([i * 5 for i in range(15)])
    # train_acc_max=np.argmax(train_acc)
    # show_max = str(train_acc[train_acc_max])[0:5]
    # plt.plot(train_acc_max,train_acc[train_acc_max], 'ko') 
    # plt.annotate(show_max,xy=(train_acc_max,train_acc[train_acc_max]),xytext=(train_acc_max,train_acc[train_acc_max]))
    annotate_max(train_acc)
    plt.plot(epoch, train_acc, color='blue', label='train')
    annotate_max(test_acc)
    plt.plot(epoch, test_acc, color='red', label='test')
    plt.legend(loc='upper left')

    plt.subplot(122)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epoch, train_loss, color='blue', label='train')
    plt.legend(loc='upper left')
    plt.savefig('loss_{}.jpg'.format(opt))