#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :gzsl_three_layer.py
@Date    :2021/05/30 21:52:25
@Author  :Chong
    在gzsl_multi.py基础上，使用两层父类
'''
from inspect import getframeinfo
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from matplotlib import pyplot as plt
from torch.nn.modules.loss import HingeEmbeddingLoss
from random import randint
import json
import argparse
import sys
sys.path.append('/home/huangchong/work/test_space/Hyperbolic_ZSL-master/poincare-embeddings')
sys.path.append('/home/huangchong/work/test_space/Hyperbolic_ZSL-master/')
import geoopt.geoopt.optim.rsgd as rsgd_
import geoopt.geoopt.optim.radam as radam_
from hyrnn.hyrnn.nets import MobiusLinear
from geoopt.geoopt.tensor import ManifoldParameter
from geoopt.geoopt.manifolds.poincare import PoincareBall
from hype.poincare import PoincareManifold as PM
from tqdm import tqdm
from sklearn import preprocessing
import numpy as np
from scipy import io, spatial
import time
from random import shuffle
import random
from sklearn.metrics import confusion_matrix
#from models.poincare_network import Image_Transformer, MyHingeLoss
import copy
# from torch.utils import data
from PIL import Image
sys.path.append('/home/huangchong/work/test_space/Hierachical_Feature_Learning/reconstruct')
import torchvision.transforms as transforms
from three_layer.dataset import Three_Layer_Dataset
from three_layer.model import GZSL_Three_Layer_HFLN
from tensorboardX import SummaryWriter
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class ClassAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_labels):
        self.reset(n_labels)

    def reset(self,n_labels):
        self.n_labels = n_labels
        self.acc = torch.zeros(n_labels)
        self.cnt = torch.Tensor([1e-8]*n_labels)
        self.pred_prob = []

    def update(self, val, cnt):
        self.acc += val
        self.cnt += cnt
        self.avg = 100*self.acc.dot(1.0/self.cnt).item()/self.n_labels # 计算所有类别的平均准确率
        #print ('pred',len(self.pred_prob))

def parse_option(): 

    parser = argparse.ArgumentParser('argument for training')

    opt = parser.parse_args()
    opt.early_stop = 10
    opt.dataset = 'AWA2'
    opt.image_dir = '/home/huangchong/work/data/AwA2/Animals_with_Attributes2/'
    opt.epochs = 150
    opt.dimension = 300
    opt.threshold = 0.8
    opt.parts = 10
    opt.print_freq = 20
    '''
        embedding_type说明用的语义embedding是属性信息，还是poincare_embedding
    '''
    opt.embedding_type = 'attribute'
    # opt.embedding_type = 'concat_embedding'
    opt.optimizer = 'sgd_steplr'
    return opt


def normalizeFeature(x):
    # x = N x d (d:feature dimension, N:number of instances)
    x = x + 1e-10
    feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
    feat = x / feature_norm[:, np.newaxis]
    return feat
def gzsl_multi_evaluate(config, model, data, n_labels, origin, father_n_labels, fa_origin, top_n_labels, top_origin):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    # n_labels = sig.shape[0]
    # print('------------gzsl_multi_evaluate:n_labels:', n_labels)
    # father_n_labels = father_sig.shape[0]
    # print('------------gzsl_multi_evaluate:father_n_labels:', father_n_labels)
    class_avg = ClassAverageMeter(n_labels)
    father_top1 = AverageMeter()
    father_avg = ClassAverageMeter(father_n_labels)
    top_top1 = AverageMeter()
    top_avg = ClassAverageMeter(top_n_labels)
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, father_label, top_label) in enumerate(data):

            input = input.cuda(non_blocking=True)
            target = target.long().cuda(non_blocking=True)
            father_label = father_label.long().cuda(non_blocking=True) # [64]
            top_label = top_label.long().cuda(non_blocking=True) # [64]

            output, father_output, top_output = model(input)

            _, pred = output.topk(1, 1, True, True)
            _, father_pred = father_output.topk(1, 1, True, True)
            _, top_pred = top_output.topk(1, 1, True, True)

            bs = input.size(0)
            prec1,class_acc,class_cnt = gzsl_multi_accuracy(pred, target, bs, n_labels, origin)
            father_prec1, father_acc, father_cnt = gzsl_multi_accuracy(father_pred, father_label,  bs, father_n_labels, fa_origin)
            top_prec1, top_acc, top_cnt = gzsl_multi_accuracy(top_pred, top_label, bs, top_n_labels, top_origin)
            father_top1.update(father_prec1, bs)
            father_avg.update(father_acc, father_cnt)

            top_top1.update(top_prec1, bs)
            top_avg.update(top_acc, top_cnt)

            top1.update(prec1, bs)
            class_avg.update(class_acc,class_cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                      'Class avg {class_avg.avg:.3f} '.format(
                    i, len(data), batch_time=batch_time,
                    class_avg=class_avg, top1=top1))
                print('\tmid prec@1 {top1.val:.3f} mid avg {top1.avg:.3f}'
                    'mid Class Avg {lbl_avg.avg:.3f}'.format(top1=father_top1, lbl_avg=father_avg))
                # print(top_avg.avg, top_top1.val, top_top1.avg)
                print('\ttop prec@1 {top1.val:.3f} top avg {top1.avg:.3f}'
                    'top Class Avg {lbl_avg.avg:.3f}'.format(top1=top_top1, lbl_avg=top_avg))
    return class_avg.avg, father_avg.avg, top_avg.avg
def accuracy(pred, target, batch_size, n_labels):
    correct = 0
    class_accuracy = torch.zeros(n_labels)
    class_cnt = torch.zeros(n_labels)
    for i in range(batch_size):
        t = target[i] # 第i个样本对应的真实label
        if pred[i] == t:
            correct += 1
            class_accuracy[t] += 1
        class_cnt[t] += 1
    return correct*100.0 / batch_size, class_accuracy, class_cnt
def gzsl_multi_accuracy(pred, target, batch_size, n_labels, origin):
    correct = 0
    class_accuracy = torch.zeros(n_labels)
    class_cnt = torch.zeros(n_labels)
    for i in range(batch_size):
        # 对于test_unseen来讲，t属于[0, 9]，而gzsl设置下，pred[i]属于[0,49]，因此将t转化为原始编号下的标签tt。
        t = target[i] # 第i个样本对应的真实label
        tt = origin[t.item()]
        # print('tt:', tt)
        tt = torch.tensor(tt).long().cuda()
        # print(pred[i].type(), tt.type())
        # print('tt:', tt, tt.type())
        if pred[i] == tt:
            correct += 1
            class_accuracy[t] += 1
        class_cnt[t] += 1
    return correct*100.0 / batch_size, class_accuracy, class_cnt
def wrap_data(args, trainval_img, test_img, test_seen_img, labels_trainval, labels_test, labels_test_seen, trainval_embedding, test_embedding, test_seen_embedding, father_labels_train, father_labels_test, father_labels_test_seen):
    '''
        将原始数据用Dataset、Dataloader包装起来以供模型读取
    '''
    params = {
        'batch_size': 64,
        # 'batch_size': 32,
        'num_workers': 4, 
        'pin_memory': True,
        'sampler': None
    }
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    '''
        2021-05-12日测试，输入图像变成448*448的效果
    '''
    tr_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.08,1),(0.5,4.0 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    ts_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    trainval_data = Multi_Dataset(args, trainval_img, labels_trainval, trainval_embedding, tr_transforms, father_labels_train)
    test_data = Multi_Dataset(args, test_img, labels_test, test_embedding, ts_transforms, father_labels_test)
    test_seen_data = Multi_Dataset(args, test_seen_img, labels_test_seen, test_seen_embedding, ts_transforms, father_labels_test_seen)
    print('trainval_data:{},test_data:{}'.format(len(trainval_data), len(test_data)))
    params['shuffle'] = True
    params['sampler'] = None
    trainval_data = data.DataLoader(trainval_data, **params)
    params['shuffle'] = False
    test_data = data.DataLoader(test_data, **params)
    test_seen_data = data.DataLoader(test_seen_data, **params)
    return trainval_data, test_data, test_seen_data
def get_father_label(mode, labels_trainval, labels_test, labels_test_seen):
    '''
        mode即选择获得mid层父类的还是top层父类的
        mode = mid or top
    '''
    son2father = {}
    with open('/home/huangchong/work/test_space/project/concat_result/attribute/three_layer_result/son2'+mode+'.txt', 'r') as f:
        son2father = eval(f.read())
    print('son2'+mode+':', son2father)
    father_labels_train = np.array(list(map(lambda x: son2father[x - 1], labels_trainval)))
    father_labels_test = np.array(list(map(lambda x: son2father[x - 1], labels_test)))
    father_labels_test_seen = np.array(list(map(lambda x: son2father[x - 1], labels_test_seen)))
    father_labels = {}
    father_labels['train'] = father_labels_train
    father_labels['test'] = father_labels_test
    father_labels['test_seen'] = father_labels_test_seen
    return father_labels
def map_labels(mid_labels):
    '''
        将labels映射到连续标签，并返回：
            可见类连续标签对应的原始类别
            不可见类连续标签对应的原始类别
            映射后的标签
    '''
    mid_seen2origin = {}
    i = 0
    for label in mid_labels['seen']:
        mid_labels['train'][ mid_labels['train'] == label ] = i
        mid_seen2origin[i] = label
        i +=1
    
    mid_unseen2origin = {}
    j = 0
    for label in mid_labels['unseen']:
        mid_labels['test'][ mid_labels['test'] == label ] = j
        mid_unseen2origin[j] = label
        j += 1
    k = 0
    for label in mid_labels['seen']:
        mid_labels['test_seen'][ mid_labels['test_seen'] == label ] = k
        k += 1
    return mid_labels, mid_seen2origin, mid_unseen2origin
def load_attr_vec(mode, labels):
    '''
        mode: mid or top
    '''
    sig = np.load('/home/huangchong/work/test_space/project/concat_result/attribute/three_layer_result/'+mode+'_attr_vec.npy')
    attr = {}
    attr['all'] = sig
    print('{}_attr:{}'.format(mode, sig.shape))
    attr['seen'] = sig[labels['seen'], :]
    attr['unseen'] = sig[labels['unseen'], :]
    return attr
def get_gzsl_data(args):
    '''
        获取zsl setting下的训练数据和测试数据
    '''
    data_folder = '/home/huangchong/work/Popular-ZSL-Algorithms/xlsa17/data/'+args.dataset+'/'
    res101 = io.loadmat(data_folder+'res101.mat')
    att_splits=io.loadmat(data_folder+'att_splits.mat')
    image_files = np.load(data_folder+'image_files.npy')
    trainval_loc = 'trainval_loc'
    test_loc = 'test_unseen_loc'
    test_seen_loc = 'test_seen_loc'

    # train:16187 val:7340 tr+val:23527 test seen:5882 test unseen:7913
    trainval_img = image_files[np.squeeze(att_splits[trainval_loc]-1)]
    test_img = image_files[np.squeeze(att_splits[test_loc]-1)]
    test_seen_img = image_files[np.squeeze(att_splits[test_seen_loc]-1)]

    print('TrainVal:{}; Ts:{}\n'.format(trainval_img.shape[0], test_img.shape[0]))
    labels = res101['labels'] # [37322, 1] 类别下标
    labels_trainval = np.squeeze(labels[np.squeeze(att_splits[trainval_loc]-1)]) # (23527,) 训练图像的label，下标从1开始的
    labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])
    labels_test_seen = np.squeeze(labels[np.squeeze(att_splits[test_seen_loc]-1)])
    labels = {}
    labels['train'] = labels_trainval - 1
    labels['test'] = labels_test - 1
    labels['test_seen'] = labels_test_seen - 1
    '''
        获得父类的标签
    '''    
    mid_labels = get_father_label('mid', labels_trainval, labels_test, labels_test_seen)
    top_labels = get_father_label('top', labels_trainval, labels_test, labels_test_seen)
    print('mid_labels train/test test_seen:', len(mid_labels['train']), len(mid_labels['test']), len(mid_labels['test_seen']))
    print('top_labels train/test test_seen:', len(top_labels['train']), len(top_labels['test']), len(top_labels['test_seen']))

    # father_labels_test =
    '''
        获得父类的unique标签，分可见类和不可见类
        TODO 这部分可以放到get_father_label中完成
    '''
    mid_labels['seen'] = np.unique(mid_labels['train'])
    # father_train_seen = np.unique(mid_labels['train'])
    print('mid_seen unique:', mid_labels['seen'])
    mid_labels['unseen'] = np.unique(mid_labels['test'])
    print('mid_unseen unique:', mid_labels['unseen'])

    top_labels['seen'] = np.unique(top_labels['train'])
    print('top_seen unique:', top_labels['seen'])
    top_labels['unseen'] = np.unique(top_labels['test'])
    print('top_unseen unique:', top_labels['unseen'])

    '''
        TODO 有待进一步修改，可以考虑用杨再权那种方式
        这里是将离散的标签转为连续的，以用于训练，以及用于不可见类的分类
        seen2origin是从连续标签到离散标签的映射
    '''
    mid_labels, mid_seen2origin, mid_unseen2origin = map_labels(mid_labels)
    top_labels, top_seen2origin, top_unseen2origin = map_labels(top_labels)
    print('---------------------------查看label是否映射正确--------------------')
    print(mid_labels['train'].min(), mid_labels['train'].max())
    print(mid_labels['test'].min(), mid_labels['test'].max())
    # print(np.unique(father_labels_train), np.unique(father_labels_test))
    print('mid_seen2origin:', mid_seen2origin)
    print('mid_unseen2origin:', mid_unseen2origin)
    print('top_seen2origin:', top_seen2origin)
    print('top_unseen2origin:', top_unseen2origin)

    labels['seen'] = np.unique(labels['train']) # (27,) 下标从1开始，所以下面算train_sig时才会减1
    print('labels_seen:', labels['seen'])
    labels['unseen'] = np.unique(labels['test']) # (27,) 下标从1开始，所以下面算train_sig时才会减1
    print('labels_unseen:', labels['unseen'])
    #  做的应该是把不连续的标签，转化成连续的，如0-27之间的。
    # seen2origin是将连续化后的标签映射为原本此标签在0~49中的编号（因为gzsl测试时对50类进行预测）
    labels, seen2origin, unseen2origin = map_labels(labels)
    print('seen2origin:', seen2origin)
    print('unseen2origin:', unseen2origin)
    classes = {}
    classes['train'] = np.unique(labels_trainval)
    #  TODO 此处可以用参数控制使用哪种信息，使用属性信息时不需要归一化，因为是归一化好的。
    if args.embedding_type == 'attribute':
        sig = att_splits['att'].T # (85, 50)->(50, 85)
        assert sig.shape == (50, 85)
        # father_sig已经是做完l2归一化的，因此不需要再归一化了
        mid_attr = load_attr_vec('mid', mid_labels)
        top_attr = load_attr_vec('top', top_labels)
        print('mid_train/test_attr.shape:', mid_attr['seen'].shape, mid_attr['unseen'].shape)
        print('train_train/test_attr.shape:', top_attr['seen'].shape, top_attr['unseen'].shape)
    else:
        sig = np.load('/home/huangchong/work/test_space/Hierachical_Feature_Learning/multi_branch/vec_array.npy') # (50, 300)
        sig = normalizeFeature(sig)
        assert sig.shape == (50, 200)
        father_sig = np.load('/home/huangchong/work/test_space/Hierachical_Feature_Learning/multi_branch/father_vec_array.npy')
        father_sig = normalizeFeature(father_sig)
        print('father_sig:', father_sig.shape)
        father_train_sig = father_sig[father_train_seen, :]
        father_test_sig = father_sig[father_test_unseen, :]
        father_test_seen_sig = father_sig[father_test_seen, :]
        print('father_train/test_sig.shape:', father_sig.shape, father_train_sig.shape, father_test_sig.shape)
    print('sig:', sig.shape)
        
    # Shape -> (Number of attributes, Number of Classes)
    # 12-16 (27, 300) 不取.T
    attr = {}
    attr['all'] = sig
    attr['seen'] = sig[labels['seen'], :]
    attr['unseen'] = sig[labels['unseen'], :]
    attr['train'] = attr['seen'][labels['train'], :]
    attr['test'] = attr['unseen'][labels['test'], :]
    attr['test_seen'] = attr['seen'][labels['test_seen'], :]

    print('trainval_embd:{},test_embd:{}'.format(attr['train'].shape, attr['test'].shape))
    img = {}
    img['train'] = trainval_img
    img['test'] = test_img
    img['test_seen'] = test_seen_img
    '''
        img: 
            train, test, test_seen
        labels:
            train, test, test_seen
            seen, unseen
        mid_labels/top_labels:
            字段同labels
        attr:
            all, train, test, test_seen
            seen, unseen
        mid_attr/top_attr:
            all, seen, unseen
    '''
    '''
        将原始数据用Dataset、Dataloader包装起来以供模型读取
    '''
    params = {
        'batch_size': 64,
        # 'batch_size': 32,
        'num_workers': 4, 
        'pin_memory': True,
        'sampler': None
    }
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    tr_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.08,1),(0.5,4.0 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    ts_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    data = {}
    data['train'] = Three_Layer_Dataset(args, img['train'], labels['train'], tr_transforms, mid_labels['train'], top_labels['train'])
    data['test'] = Three_Layer_Dataset(args, img['test'], labels['test'], ts_transforms, mid_labels['test'], top_labels['test'])
    data['test_seen'] = Three_Layer_Dataset(args, img['test_seen'], labels['test_seen'], ts_transforms, mid_labels['test_seen'], top_labels['test_seen'])
    print('dataset len: trainval_data:{},test_data:{}'.format(len(data['train']), len(data['test'])))
    params['shuffle'] = True
    params['sampler'] = None
    data['train'] = torch.utils.data.DataLoader(data['train'], **params)
    params['shuffle'] = False
    data['test'] = torch.utils.data.DataLoader(data['test'], **params)
    data['test_seen'] = torch.utils.data.DataLoader(data['test_seen'], **params)
    '''
        data: train, test, seen
        attr: all, seen, unseen
    '''
    map_dict = {}
    map_dict['seen'] = seen2origin
    map_dict['unseen'] = unseen2origin
    map_dict['mid_seen'] = mid_seen2origin
    map_dict['mid_unseen'] = mid_unseen2origin
    map_dict['top_seen'] = top_seen2origin
    map_dict['top_unseen'] = top_unseen2origin
    return data, attr, mid_attr, top_attr, classes, map_dict
    # trainval_data, test_data, test_seen_data = wrap_data(args, trainval_img, test_img, test_seen_img, labels_trainval, labels_test, labels_test_seen, trainval_embedding, test_embedding, test_seen_embedding, father_labels_train, father_labels_test, father_labels_test_seen)
    # return trainval_data, test_data, test_seen_data, trainval_sig, sig, trainval_classes, father_train_sig, father_sig, seen2origin, unseen2origin, fa_seen2origin, fa_unseen2origin
def save_model(model, optimizer, epoch, best_acc, best_epoch, fname, is_best):
    state = {
        'epoch': epoch+1,
        'arch': 'resnet101',
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_HM': best_acc,
        'best_epoch': best_epoch,
    }
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, './three_layer/multi_models/steplr3_weight_decay_AWA2_PS_model_best.pth.tar')
    return 
import sys
if __name__ == '__main__':
    args = parse_option()
    random.seed(42)
    np.random.seed()
    data, attr, mid_attr, top_attr, classes, map_dict = get_gzsl_data(args)
    writer = SummaryWriter('visualize/three_layer_gzsl/multi_steplr3_wd')

    print('--------ZSL setting-----------\n')
    print('train_data:{},test_unseen_data:{},test_seen_data:{}'.format(len(data['train']), len(data['test']), len(data['test_seen'])))


    model = models.resnet101(pretrained=True)
    '''
        attr:
            seen, unseen, all
    '''
    model = GZSL_Three_Layer_HFLN(args, model, attr, mid_attr, top_attr)
    model = nn.DataParallel(model).cuda()
    # TODO 去除了对线性层的norm
    # for m in model.modules():
    #     if isinstance(m, nn.Linear):
    #         m.weight.data.normal_()
    criterion = nn.CrossEntropyLoss().cuda()
    # TODO 可以加上monmentum和weight decay
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    elif args.optimizer == 'sgd_steplr':
        # sgd_steplr 1 : 0.001 20 0.1
        # sgd_steplr 2 : 0.001 30 0.1
        # sgd_steplr 3 : 0.001 90 0.5
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=best_cfg['lr'],
    #                                  momentum=best_cfg['momentum'],
    #                                  weight_decay=best_cfg['weight_decay'])
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    best_tr_acc = 0.0
    best_tr_ep = -1

    
    n_train_labels = attr['seen'].shape[0]
    father_n_train_labels = mid_attr['seen'].shape[0]
    top_n_train_labels = top_attr['seen'].shape[0]

    best_epoch = -1
    best_acc = -1
    best_father_acc = -1
    best_father_epoch = -1
    best_top_acc = -1
    best_top_epoch = -1
    
    iter_num = len(data['train'])
    for ep in range(args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        class_avg = ClassAverageMeter(n_train_labels)

        father_losses = AverageMeter()
        father_top1 = AverageMeter()
        father_avg = ClassAverageMeter(father_n_train_labels)

        top_losses = AverageMeter()
        top_top1 = AverageMeter()
        top_avg = ClassAverageMeter(top_n_train_labels)

        model.train()
        end = time.time()
        cnt = 0
        print('Epoch:{}'.format(ep))
        for i, (input, label, father_label, top_label) in enumerate(data['train']):
            # data:torch.FloatTensor label:torch.ByteTensor train_sig numpy
            data_time.update(time.time()-end)
            # embd = torch.from_numpy(embd).float().cuda()
            input = input.cuda(non_blocking=True) # [64,3,224,224]
            label = label.long().cuda(non_blocking=True) # [64]
            father_label = father_label.long().cuda(non_blocking=True) # [64]
            top_label = top_label.long().cuda(non_blocking=True)

            bs = label.shape[0]
            output, father_output, top_output = model(input)
            _, pred = output.topk(1, 1, True, True)
            _, father_pred = father_output.topk(1, 1, True, True)
            _, top_pred = top_output.topk(1, 1, True, True)
            loss = criterion(output, label)
            father_loss = criterion(father_output, father_label)
            top_loss = criterion(top_output, top_label)
            prec1, class_acc, class_cnt = accuracy(pred, label, bs, n_train_labels)

            father_prec1, father_acc, father_cnt = accuracy(father_pred, father_label,  bs, father_n_train_labels)
            father_losses.update(father_loss.item(), bs)
            father_top1.update(father_prec1, bs)
            father_avg.update(father_acc, father_cnt)

            top_prec1, top_acc, top_cnt = accuracy(top_pred, top_label, bs, top_n_train_labels)
            top_losses.update(top_loss.item(), bs)
            top_top1.update(top_prec1, bs)
            top_avg.update(top_acc, top_cnt)
            
            loss = top_loss + father_loss + loss

            losses.update(loss.item(), bs)
            top1.update(prec1, bs)
            class_avg.update(class_acc, class_cnt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}] '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                    'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                    'Class avg {lbl_avg.avg:.3f} '.format(
                    ep, i, iter_num, batch_time=batch_time,
                    data_time=data_time, loss=losses, lbl_avg=class_avg, top1=top1))
                # TODO 报错:IndexError: too many indices for tensor of dimension 4 但可以正常训练
                print('\t mid Loss {loss.val:.3f} mid Loss Avg {loss.avg}'
                    'mid Class Avg {lbl_avg.avg:.3f}'.format(loss=father_losses, lbl_avg=father_avg))
                print('\t top Loss {loss.val:.3f} top Loss Avg {loss.avg}'
                    'top Class Avg {lbl_avg.avg:.3f}'.format(loss=top_losses, lbl_avg=top_avg))
                
                
        if args.optimizer == 'sgd_steplr':
            scheduler.step()  
        writer.add_scalar('loss/train_loss', losses.avg, ep)
        writer.add_scalar('loss/mid_train_loss', father_losses.avg, ep)
        writer.add_scalar('loss/top_train_loss', top_losses.avg, ep)
        writer.add_scalar('acc/train_acc', class_avg.avg, ep)
        writer.add_scalar('mid_acc/mid_train_acc', father_avg.avg, ep)
        writer.add_scalar('top_acc/top_train_acc', top_avg.avg, ep)
        
        # train_loss_list.append(losses.avg)
        # train_acc_list.append(class_avg.avg)
        
        print('Testing...\n')
        model.eval()
        test_acc, father_test_acc, top_test_acc = gzsl_multi_evaluate(args, model, data['test'], n_labels=10, origin=map_dict['unseen'], father_n_labels=7, fa_origin=map_dict['mid_unseen'], top_n_labels=5, top_origin=map_dict['top_unseen'])
        test_seen_acc, father_test_seen_acc, top_test_seen_acc = gzsl_multi_evaluate(args, model, data['test_seen'], n_labels=40, origin=map_dict['seen'], father_n_labels=17, fa_origin=map_dict['mid_seen'], top_n_labels=8, top_origin=map_dict['top_seen'])
        writer.add_scalar('acc/test_acc', test_acc, ep)
        writer.add_scalar('mid_acc/mid_test_acc', father_test_acc, ep)
        writer.add_scalar('top_acc/top_test_acc', top_test_acc, ep)

        writer.add_scalar('acc/test_seen_acc', test_seen_acc, ep)
        writer.add_scalar('mid_acc/mid_test_seen_acc', father_test_seen_acc, ep)
        writer.add_scalar('top_acc/top_test_seen_acc', top_test_seen_acc, ep)

        HM = 2 * test_acc * test_seen_acc / (test_acc + test_seen_acc)
        father_HM = 2 * father_test_acc * father_test_seen_acc / (father_test_acc + father_test_seen_acc)
        top_HM = 2 * top_test_acc * top_test_seen_acc / (top_test_acc + top_test_seen_acc)
        writer.add_scalar('acc/HM', HM, ep)
        writer.add_scalar('mid_acc/mid_HM', father_HM, ep)
        writer.add_scalar('top_acc/top_HM', top_HM, ep)
        is_best = HM > best_acc
        if is_best:
            best_epoch = ep + 1
            best_acc = HM
            save_model(model, optimizer, ep, best_acc, best_epoch, './three_layer/multi_models/steplr3_wd_AWA2_PS_epoch{}_HM{}_checkpoint.pth.tar'.format(best_epoch, best_acc), is_best)
        if (father_HM > best_father_acc):
            best_father_epoch = ep + 1
            best_father_acc = father_HM
        if top_HM > best_top_acc:
            best_top_epoch = ep + 1
            best_top_acc = top_HM
        # early_stop = 100
        # if ep + 1 - best_epoch > early_stop:
        #     print('Early Stopping @ Epoch:{}'.format(ep + 1))
        #     break
        print('\nBest HM:{}@ Epoch {} Best Father HM:{}@ Epoch {} Best Top HM:{}@ Epoch {}'.format(best_acc, best_epoch, best_father_acc, best_father_epoch, best_top_acc, best_top_epoch))
