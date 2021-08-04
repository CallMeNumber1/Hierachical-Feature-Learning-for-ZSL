#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :multi_main.py
@Date    :2021/01/10 19:47:02
@Author  :Chong
重构多分支main.py，差别是使用属性embedding信息
'''
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
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
from image_dataset import Multi_Dataset
from hfln import Multi_HFLN
from tensorboardX import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
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

    # parser.add_argument('--model_folder', type=str, default='./results/model/devise')
    # parser.add_argument('--loss_path', type=str, default='./results/loss/loss_test.jpg')
    # parser.add_argument('--train_img_mat_path', type=str, default='../../data/train/img/train_image_mat_resnet50.npy')
    # parser.add_argument('--val_img_mat_path', type=str, default='../../data/val/img/val_image_mat_resnet50.npy')
    # parser.add_argument('--word_model', type=str, default='glove')
    
    # parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    # parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    # parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    # parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    # parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    # parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    # parser.add_argument('--mode', type=str, default='normal')
    # parser.add_argument('--dimension', type=int, default='dimension')
    opt = parser.parse_args()

	# data
    # if opt.word_model == 'glove':
    #     pass

    # if not os.path.isdir(opt.model_folder): 
    #     os.makedirs(opt.model_folder)

    # print('model:%s mode:%s folder:%s loss:%s dimension:%s' %(opt.word_model, opt.mode, opt.model_folder, opt.loss_path, str(opt.dimension)))
    opt.early_stop = 10
    opt.dataset = 'AWA2'
    opt.image_dir = '/home/huangchong/work/data/AwA2/Animals_with_Attributes2/'
    opt.epochs = 100
    opt.dimension = 300
    opt.threshold = 0.8
    opt.parts = 10
    opt.print_freq = 20
    opt.embedding_type = 'attribute'
    opt.optimizer = 'sgd_steplr'
    return opt
# TODO pytorch 2范数归一化


def set_model(args): 
    # model     = Image_Transformer(args.dimension)
    # TODO 初始化模型参数
    # for m in model.modules():
    #     if isinstance(m, MobiusLinear):
    #         # print('before:', m.weight.data)
    #         m.weight.data.normal_()
    #         # print('after:', m.weight.data)
    # criterion = RankingLoss(0.5)
    return model, criterion

def set_optimizer(args,model): 
    # TODO 这里参数可以调参
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1) # 0.002->0.0002->0.02
    # optimizer = radam_.RiemannianAdam(model.parameters(), lr=0.01, stabilize=10)
    optimizer = radam_.RiemannianAdam(model.parameters(), lr=0.1, stabilize=10)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # 0.002->0.0002->0.02
    return optimizer

def normalizeFeature(x):
    # x = N x d (d:feature dimension, N:number of instances)
    x = x + 1e-10
    feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
    feat = x / feature_norm[:, np.newaxis]
    return feat
def multi_evaluate(config, model, criterion, data, sig, father_sig):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    n_labels = sig.shape[0]
    father_n_labels = father_sig.shape[0]
    class_avg = ClassAverageMeter(n_labels)
    father_losses = AverageMeter()
    father_top1 = AverageMeter()
    father_avg = ClassAverageMeter(father_n_labels)
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, embd, father_label) in enumerate(data):

            input = input.cuda(non_blocking=True)
            target = target.long().cuda(non_blocking=True)
            father_label = father_label.long().cuda(non_blocking=True) # [64]

            output, father_output = model(input)
            loss = criterion(output, target)
            father_loss = criterion(father_output, father_label)

            _, pred = output.topk(1, 1, True, True)
            _, father_pred = father_output.topk(1, 1, True, True)

            bs = input.size(0)
            prec1,class_acc,class_cnt = accuracy(pred, target, bs, n_labels)
            father_prec1, father_acc, father_cnt = accuracy(father_pred, father_label,  bs, father_n_labels)
            father_losses.update(father_loss.item(), bs)
            father_top1.update(father_prec1, bs)
            father_avg.update(father_acc, father_cnt)

            loss = father_loss + loss

            losses.update(loss.item(), bs)
            top1.update(prec1, bs)
            class_avg.update(class_acc,class_cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Loss {loss.val:.4f} (avg: {loss.avg:.4f}) '
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f}) '
                      'Class avg {class_avg.avg:.3f} '.format(
                    i, len(data), batch_time=batch_time, loss=losses,
                    class_avg=class_avg, top1=top1))
                print('Test: [{0}/{1}] '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'father_Loss {loss.val:.4f} (father_avg: {loss.avg:.4f}) '
                    'father_Prec@1 {top1.val:.3f} (father_avg: {top1.avg:.3f}) '
                    'father Class avg {lbl_avg.avg:.3f} '.format(
                    i, len(data), batch_time=batch_time,
                    loss=father_losses, lbl_avg=father_avg, top1=father_top1))
    return class_avg.avg, father_avg.avg
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
def wrap_data(args, trainval_img, test_img, labels_trainval, labels_test, trainval_embedding, test_embedding, father_labels_train, father_labels_test):
    '''
        将原始数据用Dataset、Dataloader包装起来以供模型读取
    '''
    params = {
        'batch_size': 64,
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
    
    trainval_data = Multi_Dataset(args, trainval_img, labels_trainval, trainval_embedding, tr_transforms, father_labels_train)
    test_data = Multi_Dataset(args, test_img, labels_test, test_embedding, ts_transforms, father_labels_test)
    print('trainval_data:{},test_data:{}'.format(len(trainval_data), len(test_data)))
    params['shuffle'] = True
    params['sampler'] = None
    trainval_data = data.DataLoader(trainval_data, **params)
    params['shuffle'] = False
    test_data = data.DataLoader(test_data, **params)
    return trainval_data, test_data
def get_zsl_data(args):
    '''
        获取zsl setting下的训练数据和测试数据
    '''
    data_folder = '/home/huangchong/work/Popular-ZSL-Algorithms/xlsa17/data/'+args.dataset+'/'
    res101 = io.loadmat(data_folder+'res101.mat')
    att_splits=io.loadmat(data_folder+'att_splits.mat')
    image_files = np.load(data_folder+'image_files.npy')
    trainval_loc = 'trainval_loc'
    test_loc = 'test_unseen_loc'

    # train:16187 val:7340 tr+val:23527 test seen:5882 test unseen:7913
    trainval_img = image_files[np.squeeze(att_splits[trainval_loc]-1)]
    test_img = image_files[np.squeeze(att_splits[test_loc]-1)]

    print('TrainVal:{}; Ts:{}\n'.format(trainval_img.shape[0], test_img.shape[0]))
    labels = res101['labels'] # [37322, 1] 类别下标
    labels_trainval = np.squeeze(labels[np.squeeze(att_splits[trainval_loc]-1)]) # (23527,) 训练图像的label，下标从1开始的
    labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])

    son2father = {}
    with open('/home/huangchong/work/test_space/project/concat_result/attribute/result/son2father.txt', 'r') as f:
        son2father = eval(f.read())
    print('son2father:', son2father)
    father_labels_train = np.array(list(map(lambda x: son2father[x - 1], labels_trainval)))
    father_labels_test = np.array(list(map(lambda x: son2father[x - 1], labels_test)))
    print('father_labels train/test', len(father_labels_train), len(father_labels_test))
    # father_labels_test =  
    father_train_seen = np.unique(father_labels_train)
    print('father_train_seen:', father_train_seen)
    father_test_unseen = np.unique(father_labels_test)
    print('father_test_unseen:', father_test_unseen)
    i = 0
    for labels in father_train_seen:
        father_labels_train[father_labels_train == labels] = i
        i +=1
    j = 0
    for labels in father_test_unseen:
        father_labels_test[father_labels_test == labels] = j
        j += 1
    # print(np.unique(father_labels_train), np.unique(father_labels_test))

    trainval_labels_seen = np.unique(labels_trainval) # (27,) 下标从1开始，所以下面算train_sig时才会减1
    print('trainval_labels_seen:', trainval_labels_seen)
    test_labels_unseen = np.unique(labels_test)
    print('test_labels_seen:', test_labels_unseen)
     #  做的应该是把不连续的标签，转化成连续的，如0-27之间的。
    i=0
    for labels in trainval_labels_seen:
        labels_trainval[labels_trainval == labels] = i    
        i+=1
    k=0
    for labels in test_labels_unseen:
        labels_test[labels_test == labels] = k
        k+=1
    trainval_classes = np.unique(labels_trainval)
    #  TODO 此处可以用参数控制使用哪种信息，使用属性信息时不需要归一化，因为是归一化好的。
    if args.embedding_type == 'attribute':
        sig = att_splits['att'].T # (85, 50)->(50, 85)
        assert sig.shape == (50, 85)
        # father_sig已经是做完l2归一化的，因此不需要再归一化了
        father_sig = np.load('/home/huangchong/work/test_space/project/concat_result/attribute/result/father_attr_vec.npy')
        print('father_sig:', father_sig.shape)
        father_train_sig = father_sig[father_train_seen, :]
        father_test_sig = father_sig[father_test_unseen, :]
        print('father_train/test_sig.shape:', father_sig.shape, father_train_sig.shape, father_test_sig.shape)
    else:
        sig = np.load('./vec_array.npy') # (50, 300)
        sig = normalizeFeature(sig)
    print('sig:', sig.shape)
        
    # Shape -> (Number of attributes, Number of Classes)
    # 12-16 (27, 300) 不取.T
    trainval_sig = sig[trainval_labels_seen-1, :]
    test_sig = sig[test_labels_unseen-1, :]
    
    trainval_embedding = trainval_sig[labels_trainval, :]
    test_embedding = test_sig[labels_test, :]
    print('trainval_embd:{},test_embd:{}'.format(trainval_embedding.shape, test_embedding.shape))
    trainval_data, test_data = wrap_data(args, trainval_img, test_img, labels_trainval, labels_test, trainval_embedding, test_embedding, father_labels_train, father_labels_test)
    return trainval_data, test_data, trainval_sig, test_sig, trainval_classes, father_train_sig, father_test_sig
if __name__ == '__main__':
    args = parse_option()
    random.seed(42)
    np.random.seed()
    train_data, test_data, train_sig, test_sig, train_classes, father_train_sig, father_test_sig = get_zsl_data(args)
    writer = SummaryWriter('visualize/multi_steplr')

    print('--------ZSL setting-----------\n')
    print('train_data:{},test_data:{}'.format(len(train_data), len(test_data)))
    
    model = models.resnet101(pretrained=True)
    model = Multi_HFLN(args, model, train_sig, test_sig, father_train_sig, father_test_sig)
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
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=best_cfg['lr'],
    #                                  momentum=best_cfg['momentum'],
    #                                  weight_decay=best_cfg['weight_decay'])

    best_tr_acc = 0.0
    best_tr_ep = -1

    n_train_labels = train_sig.shape[0]
    father_n_train_labels = father_train_sig.shape[0]

    best_epoch = -1
    best_acc = -1
    best_father_acc = -1
    best_father_epoch = -1
    for ep in range(args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        class_avg = ClassAverageMeter(n_train_labels)

        father_losses = AverageMeter()
        father_top1 = AverageMeter()
        father_avg = ClassAverageMeter(father_n_train_labels)
        model.train()
        end = time.time()
        cnt = 0
        print('Epoch:{}'.format(ep))
        for i, (data, label, embd, father_label) in enumerate(train_data):
            # data:torch.FloatTensor label:torch.ByteTensor train_sig numpy
            data_time.update(time.time()-end)
            # embd = torch.from_numpy(embd).float().cuda()
            data = data.cuda(non_blocking=True) # [64,3,224,224]
            label = label.long().cuda(non_blocking=True) # [64]
            father_label = father_label.long().cuda(non_blocking=True) # [64]
            
            bs = label.shape[0]
            output, father_output = model(data)
            _, pred = output.topk(1, 1, True, True)
            _, father_pred = father_output.topk(1, 1, True, True)
            loss = criterion(output, label)
            father_loss = criterion(father_output, father_label)
            prec1, class_acc, class_cnt = accuracy(pred, label, bs, n_train_labels)

            father_prec1, father_acc, father_cnt = accuracy(father_pred, father_label,  bs, father_n_train_labels)
            father_losses.update(father_loss.item(), bs)
            father_top1.update(father_prec1, bs)
            father_avg.update(father_acc, father_cnt)
            
            loss = father_loss + loss

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
                    ep, i, len(train_data), batch_time=batch_time,
                    data_time=data_time, loss=losses, lbl_avg=class_avg, top1=top1))
                print('Epoch: [{0}][{1}/{2}] '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'father_Loss {loss.val:.4f} (father_avg: {loss.avg:.4f}) '
                    'father_Prec@1 {top1.val:.3f} (father_avg: {top1.avg:.3f}) '
                    'father Class avg {lbl_avg.avg:.3f} '.format(
                    ep, i, len(train_data), batch_time=batch_time,
                    data_time=data_time, loss=father_losses, lbl_avg=father_avg, top1=father_top1))
        if args.optimizer == 'sgd_steplr':
            scheduler.step()  
        writer.add_scalar('loss/train_loss', losses.avg, ep)
        writer.add_scalar('loss/father_train_loss', father_losses.avg, ep)
        writer.add_scalar('acc/train_acc', class_avg.avg, ep)
        writer.add_scalar('acc/father_train_acc', father_avg.avg, ep)
        
        # train_loss_list.append(losses.avg)
        # train_acc_list.append(class_avg.avg)
        
        print('Testing...\n')
        model.eval()

        test_acc, father_test_acc = multi_evaluate(args, model, criterion, test_data, test_sig, father_test_sig)
        writer.add_scalar('acc/test_acc', test_acc, ep)
        writer.add_scalar('acc/father_test_acc', father_test_acc, ep)
        if (test_acc > best_acc):
            best_epoch = ep + 1
            best_acc = test_acc
            # save_model(model, optimizer, ep, best_acc, best_epoch, './models/AWA2_PS_epoch{}_acc{}_checkpoint.pth.tar'.format(best_epoch, best_acc))
        if (father_test_acc > best_father_acc):
            best_father_epoch = ep + 1
            best_father_acc = father_test_acc
        # early_stop = 100
        # if ep + 1 - best_epoch > early_stop:
        #     print('Early Stopping @ Epoch:{}'.format(ep + 1))
        #     break
        print('\nBest Test Acc:{}@ Epoch {} Best Father Acc:{}@ Epoch {}'.format(best_acc, best_epoch, best_father_acc, best_father_epoch))