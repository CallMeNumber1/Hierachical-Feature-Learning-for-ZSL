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
from tqdm import tqdm
import os
#os.environ['CUDA_VISIBLE_DEVISES'] = '1,2,3'
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
    opt.epochs = 200
    opt.dimension = 300
    opt.threshold = 0.8
    opt.parts = 10
    opt.print_freq = 20
    return opt
# TODO pytorch 2范数归一化


def set_model(args): 
    model     = Image_Transformer(args.dimension)
    # TODO 初始化模型参数
    # for m in model.modules():
    #     if isinstance(m, MobiusLinear):
    #         # print('before:', m.weight.data)
    #         m.weight.data.normal_()
    #         # print('after:', m.weight.data)
    # TODO 这里的margin可以调参，或者更换损失
    criterion = RankingLoss(0.5)
    # criterion = nn.L1Loss()
    return model, criterion
# class RankingLoss(torch.nn.Module):
#     def __init__(self, margin):
#         super(RankingLoss, self).__init__()
#         self.margin = margin
#     def forward(self, pred_embedding, batch_label, train_classes, train_sig):
#         # pred_emebedding:[32, 300] trainc_classes:0~27
#         idx = pred_embedding.shape[0]
#         loss = torch.tensor(0.0, requires_grad=True).cuda()
#         train_sig = torch.from_numpy(train_sig).float().cuda() # [300, 27]
#         for j in range(idx):
#             y_n = batch_label[j]
#             y_ = train_classes[train_classes != y_n]
#             y_n = torch.from_numpy(np.array(y_n)).byte().cuda() 
#             # print('y_n', y_n)
#             y_ = torch.from_numpy(y_).byte().cuda() # [26]
#             XW = pred_embedding[j] # [300]
#             # print('XW.shape:{}, train_sig.shape:{}, y_n.shape:{}, y_.shape:{}'.format(XW.shape, train_sig.shape, y_n.shape, y_.shape))
#             gt_class_score = PM().distance(XW, train_sig[:, y_n.item()])  # []
#             # print('gl_class_score.shape:', gt_class_score.shape)
#             for i, label in enumerate(y_):
#                 # loss += torch.relu(self.margin + torch.dot(XW, train_sig[:, label.item()]) - gt_class_score)
#                 tmp = (self.margin + PM().distance(XW, train_sig[:, label.item()]) - gt_class_score)
#                 if tmp > 0:
#                     loss += tmp
#         return loss
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
def poincare_distances(XW, sig):
    # sig.shape: [27, 200]
    res = np.zeros((XW.shape[0], sig.shape[0])) # N * c
    for i in range(XW.shape[0]):
        item = XW[i] # (200,)
        # print('XW[i]:{}, sig.shape:{}'.format(item.shape, sig.shape))
        for j in range(sig.shape[0]):
            item_t = torch.from_numpy(item).float().cuda()
            sigj_t = torch.from_numpy(sig[j]).float().cuda()
            # dist = 1 - PM().distance(item_t, sigj_t) # torch.cuda.FloatTensor
            dist = 1-PM().distance(item_t, sigj_t) # torch.cuda.FloatTensor
            # print('dist:{}'.format(dist.type()))
            res[i][j]=dist.item()
    return res
    # print('dist.shape:', res.shape)

def new_evaluate(config, model, criterion, data, sig):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    n_labels = sig.shape[0]
    class_avg = ClassAverageMeter(n_labels)
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, embd) in enumerate(data):

            input = input.cuda(non_blocking=True)
            target = target.long().cuda(non_blocking=True)
            output = model(input)
            loss = criterion(output, target)
            _, pred = output.topk(1, 1, True, True)
            bs = input.size(0)
            prec1,class_acc,class_cnt = accuracy(pred, target, bs, n_labels)

            losses.update(loss, bs)
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

    return class_avg.avg, top1.avg, class_avg.pred_prob

def evaluate(data, model, sig):
    avg_acc = 0
    # cnt = 0
    n_labels = sig.shape[0]
    # print('n_labels:', n_labels)
    
    top1 = AverageMeter()
    class_avg = ClassAverageMeter(n_labels)
    with torch.no_grad():
        for input, label, embd in data:
            input = input.cuda()
            XW = model(input)
            XW = XW.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            # XW = np.dot(X.T, W)# N x k
            # 第一行是第一个样本预测的结果和各个sig的距离，1-，表示求相似度，进而把找最小的变成找最大的。
            dist = 1-spatial.distance.cdist(XW, sig, 'cosine')# N x C(no. of classes)
            
            predicted_classes = np.array([np.argmax(output) for output in dist]) # (16187, )
            
            bs = input.shape[0]
            prec1, class_acc, class_cnt = accuracy(predicted_classes, label, bs, n_labels)
            top1.update(prec1, bs)
            class_avg.update(class_acc, class_cnt)
    return class_avg.avg, top1.avg
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
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
from image_dataset import Dataset
def get_data(args):
    data_folder = '/home/huangchong/work/Popular-ZSL-Algorithms/xlsa17/data/'+args.dataset+'/'
    res101 = io.loadmat(data_folder+'res101.mat')
    att_splits=io.loadmat(data_folder+'att_splits.mat')
    image_files = np.load(data_folder+'image_files.npy')
    train_loc = 'train_loc'
    val_loc = 'val_loc'
    test_loc = 'test_unseen_loc'
    feat = res101['features'] # train:16187 val:7340 tr+val:23527 test seen:5882 test unseen:7913
    train_img = image_files[np.squeeze(att_splits[train_loc]-1)]
    val_img = image_files[np.squeeze(att_splits[val_loc]-1)]
    test_img = image_files[np.squeeze(att_splits[test_loc]-1)]

    print('Tr:{}; Val:{}; Ts:{}\n'.format(train_img.shape[0], val_img.shape[0], test_img.shape[0]))
    labels = res101['labels'] # [37322, 1] 类别下标
    labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)]) # (16187,) 训练图像的label，下标从1开始的
    labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)])
    labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])
    train_labels_seen = np.unique(labels_train) # (27,) 下标从1开始，所以下面算train_sig时才会减1
    print('train_labels_seen:', train_labels_seen)
    val_labels_unseen = np.unique(labels_val)
    print('val_labels_seen:', val_labels_unseen)
    test_labels_unseen = np.unique(labels_test)
    print('test_labels_seen:', test_labels_unseen)
     #  做的应该是把不连续的标签，转化成连续的，如0-27之间的。
    i=0
    for labels in train_labels_seen:
        labels_train[labels_train == labels] = i    
        i+=1
    j=0
    for labels in val_labels_unseen:
        labels_val[labels_val == labels] = j
        j+=1
    k=0
    for labels in test_labels_unseen:
        labels_test[labels_test == labels] = k
        k+=1
    train_classes = np.unique(labels_train)
    #sig = att_splits['att'] # (85, 50)
    # sig = np.load('./googlenews_vec_array.npy') # (50, 300)
    sig = np.load('./vec_array.npy') # (50, 300)
    sig = normalizeFeature(sig)
    print('sig:', sig.shape)
        
    # Shape -> (Number of attributes, Number of Classes)
    # 12-16 (27, 300) 不取.T
    train_sig = sig[train_labels_seen-1, :]
    val_sig = sig[val_labels_unseen-1, :]
    test_sig = sig[test_labels_unseen-1, :]
    
    train_embedding = train_sig[labels_train, :]
    val_embedding =  val_sig[labels_val, :]
    test_embedding = test_sig[labels_test, :]
    print('train_embd:{},val_embd:{},test_embd:{}'.format(train_embedding.shape, val_embedding.shape, test_embedding.shape))

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
    
    train_data = Dataset(args, train_img, labels_train, train_embedding, tr_transforms)
    val_data = Dataset(args, val_img, labels_val, val_embedding, ts_transforms)
    test_data = Dataset(args, test_img, labels_test, test_embedding, ts_transforms)
    print('train_data:{},val_data:{},test_data:{}'.format(len(train_data), len(val_data), len(test_data)))
    params['shuffle'] = True
    params['sampler'] = None
    train_data = data.DataLoader(train_data, **params)
    params['shuffle'] = False
    val_data = data.DataLoader(val_data, **params)
    test_data = data.DataLoader(test_data, **params)
    return train_data, test_data, val_data, train_sig, val_sig, test_sig, train_classes, labels_train, labels_val, labels_test
class RankingLoss(torch.nn.Module):
    def __init__(self, margin):
        super(RankingLoss, self).__init__()
        self.margin = margin
    def forward(self, pred_embedding, batch_label, train_classes, train_sig):
        # pred_emebedding:[32, 300] trainc_classes:0~27
        # print('train_sig:', train_sig.shape)
        train_classes = torch.from_numpy(train_classes).byte().cuda()
        idx = pred_embedding.shape[0]
        loss = torch.tensor(0.0, requires_grad=True).cuda()
        train_sig = torch.from_numpy(train_sig).float().cuda() # [300, 27]
        for j in range(idx):
            y_n = batch_label[j]
            y_ = train_classes[train_classes != y_n]
            # y_n = torch.from_numpy(np.array(y_n)).byte().cuda() 
            # y_ = torch.from_numpy(y_).byte().cuda() # [26]
            XW = pred_embedding[j] # [300]
            # print('XW.shape:{}, train_sig.shape:{}, y_n.shape:{}, y_.shape:{}'.format(XW.shape, train_sig.shape, y_n.shape, y_.shape))
            gt_class_score = torch.dot(XW, train_sig[:, y_n.item()]) 
            for i, label in enumerate(y_):
                # loss += torch.relu(self.margin + torch.dot(XW, train_sig[:, label.item()]) - gt_class_score)
                tmp = (self.margin + torch.dot(XW, train_sig[:, label.item()]) - gt_class_score)
                if tmp > 0:
                    loss += tmp
        return loss
from hfln import NEW_HFLN
from tqdm import tqdm
if __name__ == '__main__':
    args = parse_option()
    random.seed(42)
    np.random.seed()
    train_data, test_data, val_data, train_sig, val_sig, test_sig, train_classes, labels_train, labels_val, labels_test = get_data(args)
    print('train_data:{},val_data:{},test_data:{}'.format(len(train_data), len(val_data), len(test_data)))
    
    model = models.resnet101(pretrained=True)
    # 这个特征是经过avgpool之前的，2048*7*7
    # feature_ext = nn.Sequential(*list(model.children())[:-2])
    model = NEW_HFLN(args, model, train_sig, test_sig)
    model = nn.DataParallel(model).cuda()
    # TODO 去除了对线性层的norm
    # for m in model.modules():
    #     if isinstance(m, nn.Linear):
    #         m.weight.data.normal_()
    # criterion = RankingLoss(150)
    criterion = nn.CrossEntropyLoss().cuda()
    # TODO 可以加上monmentum和weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=best_cfg['lr'],
    #                                  momentum=best_cfg['momentum'],
    #                                  weight_decay=best_cfg['weight_decay'])
    # model = model.cuda()

    # best_val_acc = 0.0
    best_tr_acc = 0.0
    # best_val_ep = -1
    best_tr_ep = -1

    n_train_labels = train_sig.shape[0]
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    class_avg = ClassAverageMeter(n_train_labels)
    model.train()
    end = time.time()
    for ep in range(args.epochs):
        model.train()
        data_time.update(time.time()-end)
        cnt = 0
        print('Epoch:{}'.format(ep))
        for i, (data, label, embd) in enumerate(train_data):
            # data:torch.FloatTensor label:torch.ByteTensor train_sig numpy
            # 貌似true-embedding没啥用
            # embd = torch.from_numpy(embd).float().cuda()
            data = data.cuda(non_blocking=True) # [64,3,224,224]
            label = label.long().cuda(non_blocking=True) # [64]
            
            bs = label.shape[0]
            output = model(data)
            _, pred = output.topk(1, 1, True, True)
            if cnt == 0:
                print('training time, label:', label)
                print('training time pred:', pred)
                print('output.shape:', output.shape, pred.shape)
            cnt += 1
            loss = criterion(output, label)
            # pred_embd = model(data)
            # loss = criterion(pred_embd, label, train_classes, train_sig.T)
            prec1, class_acc, class_cnt = accuracy(pred, label, bs, n_train_labels)
            losses.update(loss, bs)
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
        # model.eval()
        # tr_acc,_ = new_evaluate(args, model, criterion, train_data, train_sig)			
        # # val_acc,_ = evaluate(val_data, model, val_sig)
        # end = time.time()
        # elapsed = end - start
        # print('Epoch:{}; Train Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, tr_acc, elapsed//60, elapsed%60))
        # # if val_acc>best_val_acc:
        # #     best_val_acc = val_acc
        # #     best_val_ep = ep+1
        
        # if tr_acc>best_tr_acc:
        #     best_tr_ep = ep+1
        #     best_tr_acc = tr_acc
        # # 过了10个轮次 best_val仍未更新，则提前终止
        # # if ep+1-best_val_ep>args.early_stop:
        # #     print('Early Stopping by {} epochs. Exiting...'.format(args.epochs-(ep+1)))
        # #     break
        # # print('\nBest Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_tr_acc, best_tr_ep))
        # print('\nBest Train Acc:{} @ Epoch {}\n'.format(best_tr_acc, best_tr_ep))
        print('Testing...\n')
        model.eval()
        # test_acc,_ = new_evaluate(args, model, criterion, test_data, test_sig)
        new_evaluate(args, model, criterion, test_data, test_sig)
        # print('Test Acc:{}'.format(test_acc))
