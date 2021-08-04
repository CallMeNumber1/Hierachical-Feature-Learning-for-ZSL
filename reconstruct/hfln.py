'''
    我们打算做的：层次化特征学习网络：hlfn
'''
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import init
import torch
import datetime
from torch.nn.parameter import Parameter
class Multi_HFLN(nn.Module):
    '''
        多分支
    '''
    def __init__(self, cfg, model, train_attr=None, test_attr=None, father_train_attr=None, father_test_attr=None):
        super(Multi_HFLN, self).__init__()
        train_attr = Parameter(torch.from_numpy(train_attr).float().cuda())
        test_attr = Parameter(torch.from_numpy(test_attr).float().cuda())
        father_train_attr = Parameter(torch.from_numpy(father_train_attr).float().cuda())
        father_test_attr = Parameter(torch.from_numpy(father_test_attr).float().cuda())
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.father_features = self.features[0:6]

        self.cov_channel = 2048
        self.map_size = 7
        self.compress_map = 20

        self.father_cov_channel = 512
        self.father_map_size = 28
        
        f_c1, d = father_train_attr.size()
        f_c2, _ = father_test_attr.size()
        self.father_train_linear = nn.Linear(d, f_c1, False)
        self.father_train_linear.weight = father_train_attr
        for para in self.father_train_linear.parameters():
            para.requires_grad = False
        self.father_test_linear = nn.Linear(d, f_c2, False)
        self.father_test_linear.weight = father_test_attr
        for para in self.father_test_linear.parameters():
            para.requires_grad = False

        c1, d = train_attr.size()
        c2, _ = test_attr.size()
        # c3, _ = val_attr.size()
        self.train_linear = nn.Linear(d, c1, False)
        self.train_linear.weight = train_attr
        # 固定这些权重
        for para in self.train_linear.parameters():
            para.requires_grad = False
        # self.val_linear = nn.Linear(d, c3, False)
        # self.val_linear.weight = val_attr
        # for para in self.val_linear.parameters():
        #     para.requires_grad = False
        self.test_linear = nn.Linear(d, c2, False)
        self.test_linear.weight = test_attr
        for para in self.test_linear.parameters():
            para.requires_grad = False

        # aren
        self.map_threshold = cfg.threshold # 0.8
        self.parts = cfg.parts # 10
        self.map_size = 7 # Z是2048*7*7
        # TODO 这里是否
        self.pool = nn.MaxPool2d(self.map_size, self.map_size)
        self.cov = nn.Conv2d(self.cov_channel, self.parts, 1) # 2048->10
        self.p_linear = nn.Linear(self.cov_channel*self.parts, d, False) # 2048*10 -> embedding dimension
        self.dropout2 = nn.Dropout(0.4)

        self.father_pool = nn.MaxPool2d(self.father_map_size, self.father_map_size)
        self.father_cov = nn.Conv2d(self.father_cov_channel, self.parts, 1) # 512->10
        self.father_p_linear = nn.Linear(self.father_cov_channel*self.parts, d, False) # 512*10 -> embedding dimension
        self.father_dropout2 = nn.Dropout(0.4)

        self.conv_bilinear = nn.Linear(self.cov_channel, self.compress_map, 1)
        self.b_linear = nn.Linear(self.compress_map * self.cov_channel, d, False)
        self.dropout3 = nn.Dropout(0.4)
        # TODO 先冻结参数
        # for param in self.features.parameters():
        #     param.requires_grad = False
    def forward(self, x):
        father_features = self.father_features(x)
        w = father_features.size()
        weights = torch.sigmoid(self.father_cov(father_features))
        batch, parts, width, height = weights.size()
        weights_layout = weights.view(batch, -1)
        threshold_value,_ = weights_layout.max(dim=1) # [bs]
        # 找到每个Mask里面最大的
        local_max,_ = weights.view(batch,parts,-1).max(dim=2) # [bs, 10]
        # map_threshold为适应性系数
        threshold_value = self.map_threshold*threshold_value.view(batch,1) \
            .expand(batch,parts) # [bs, 10]
        # 下面这句话之后即得到经过AT之后的
        weights = weights*local_max.ge(threshold_value).view(batch,parts,1,1). \
            float().expand(batch,parts,width,height) # [bs, 10, 7, 7]
        blocks = []
        for k in range(self.parts):
            # 即[20, 2048, 7, 7] * [20, 1, 7, 7].reshape(20, 2048, 7, 7)
            Y = father_features*weights[:,k,:,:]. \
                unsqueeze(dim=1). \
                expand(w[0],self.father_cov_channel,w[2],w[3]) # [20, 2048, 7, 7]
            blocks.append(self.father_pool(Y).squeeze().view(-1,self.father_cov_channel)) # 元素为[20/22, 2048]
        # 最后这10个特征图融合了来作为最后的特征
        father_p_output = self.father_dropout2(self.father_p_linear(torch.cat(blocks, dim=1))) # torch.cat()后:[20/22, 20480]
        if self.training:
            father_p_out = self.father_train_linear(father_p_output)
        else:
            father_p_out = self.father_test_linear(father_p_output)
        
        features = self.features(x) # [20/22, 2048, 7, 7]
        w = features.size()
        # MaskGenerate # weighs.size: [20/22, 10, 7, 7]
        weights = torch.sigmoid(self.cov(features)) # # batch x parts x 7 x 7
        batch,parts,width,height = weights.size()
        weights_layout = weights.view(batch,-1) # [bs, 490]
        # 找个全局最大值
        threshold_value,_ = weights_layout.max(dim=1) # [bs]
        # 找到每个Mask里面最大的
        local_max,_ = weights.view(batch,parts,-1).max(dim=2) # [bs, 10]
        # map_threshold为适应性系数
        threshold_value = self.map_threshold*threshold_value.view(batch,1) \
            .expand(batch,parts) # [bs, 10]
        # 下面这句话之后即得到经过AT之后的
        weights = weights*local_max.ge(threshold_value).view(batch,parts,1,1). \
            float().expand(batch,parts,width,height) # [bs, 10, 7, 7]

        blocks = []
        for k in range(self.parts):
            # 即[20, 2048, 7, 7] * [20, 1, 7, 7].reshape(20, 2048, 7, 7)
            Y = features*weights[:,k,:,:]. \
                unsqueeze(dim=1). \
                expand(w[0],self.cov_channel,w[2],w[3]) # [20, 2048, 7, 7]
            blocks.append(self.pool(Y).squeeze().view(-1,self.cov_channel)) # 元素为[20/22, 2048]
        # 最后这10个特征图融合了来作为最后的特征
        p_output = self.dropout2(self.p_linear(torch.cat(blocks, dim=1))) # torch.cat()后:[20/22, 20480]
        
        if self.training:
            p_out = self.train_linear(p_output)
        else:
            p_out = self.test_linear(p_output)
            
        return p_out, father_p_out

class NEW_HFLN(nn.Module):
    '''
        单分支
    '''
    def __init__(self, cfg, model, train_attr=None, test_attr=None):
        super(NEW_HFLN, self).__init__()
        train_attr = Parameter(torch.from_numpy(train_attr).float().cuda())
        test_attr = Parameter(torch.from_numpy(test_attr).float().cuda())
        # 这个特征是经过avgpool之前的，2048*7*7
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.cov_channel = 2048
        self.map_size = 7
        self.compress_map = 20

        c1, d = train_attr.size()
        c2, _ = test_attr.size()
        # c3, _ = val_attr.size()
        self.train_linear = nn.Linear(d, c1, False)
        self.train_linear.weight = train_attr
        # 固定这些权重
        for para in self.train_linear.parameters():
            para.requires_grad = False
        # self.val_linear = nn.Linear(d, c3, False)
        # self.val_linear.weight = val_attr
        # for para in self.val_linear.parameters():
        #     para.requires_grad = False
        self.test_linear = nn.Linear(d, c2, False)
        self.test_linear.weight = test_attr
        for para in self.test_linear.parameters():
            para.requires_grad = False

        # aren
        self.map_threshold = cfg.threshold # 0.8
        self.parts = cfg.parts # 10
        self.map_size = 7 # Z是2048*7*7
        # TODO 这里是否
        self.pool = nn.MaxPool2d(self.map_size, self.map_size)
        self.cov = nn.Conv2d(self.cov_channel, self.parts, 1) # 2048->10
        self.p_linear = nn.Linear(self.cov_channel*self.parts, d, False) # 2048*10 -> embedding dimension
        self.dropout2 = nn.Dropout(0.4)

        self.conv_bilinear = nn.Linear(self.cov_channel, self.compress_map, 1)
        self.b_linear = nn.Linear(self.compress_map * self.cov_channel, d, False)
        self.dropout3 = nn.Dropout(0.4)
        # TODO 先冻结参数
        # for param in self.features.parameters():
        #     param.requires_grad = False
    def forward(self, x):
        features = self.features(x) # [20/22, 2048, 7, 7]
        w = features.size()
        # MaskGenerate # weighs.size: [20/22, 10, 7, 7]
        weights = torch.sigmoid(self.cov(features)) # # batch x parts x 7 x 7
        batch,parts,width,height = weights.size()
        weights_layout = weights.view(batch,-1) # [bs, 490]
        # 找个全局最大值
        threshold_value,_ = weights_layout.max(dim=1) # [bs]
        # 找到每个Mask里面最大的
        local_max,_ = weights.view(batch,parts,-1).max(dim=2) # [bs, 10]
        # map_threshold为适应性系数
        threshold_value = self.map_threshold*threshold_value.view(batch,1) \
            .expand(batch,parts) # [bs, 10]
        # 下面这句话之后即得到经过AT之后的
        weights = weights*local_max.ge(threshold_value).view(batch,parts,1,1). \
            float().expand(batch,parts,width,height) # [bs, 10, 7, 7]

        blocks = []
        for k in range(self.parts):
            # 即[20, 2048, 7, 7] * [20, 1, 7, 7].reshape(20, 2048, 7, 7)
            Y = features*weights[:,k,:,:]. \
                unsqueeze(dim=1). \
                expand(w[0],self.cov_channel,w[2],w[3]) # [20, 2048, 7, 7]
            blocks.append(self.pool(Y).squeeze().view(-1,self.cov_channel)) # 元素为[20/22, 2048]
        # 最后这10个特征图融合了来作为最后的特征
        p_output = self.dropout2(self.p_linear(torch.cat(blocks, dim=1))) # torch.cat()后:[20/22, 20480]
        
        if self.training:
            p_out = self.train_linear(p_output)
        else:
            p_out = self.test_linear(p_output)
            
        return p_out
class GZSL_SINGLE_HFLN(nn.Module):
    '''
        单分支
    '''
    def __init__(self, cfg, model, train_attr=None, test_attr=None):
        super(GZSL_SINGLE_HFLN, self).__init__()
        train_attr = Parameter(torch.from_numpy(train_attr).float().cuda())
        test_attr = Parameter(torch.from_numpy(test_attr).float().cuda())
        # 这个特征是经过avgpool之前的，2048*7*7
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.cov_channel = 2048
        self.map_size = 7
        self.compress_map = 20

        c1, d = train_attr.size()
        c2, _ = test_attr.size()
        # c3, _ = val_attr.size()
        self.train_linear = nn.Linear(d, c1, False)
        self.train_linear.weight = train_attr
        # 固定这些权重
        for para in self.train_linear.parameters():
            para.requires_grad = False
        self.test_linear = nn.Linear(d, c2, False)
        self.test_linear.weight = test_attr
        for para in self.test_linear.parameters():
            para.requires_grad = False
        # aren
        self.map_threshold = cfg.threshold # 0.8
        self.parts = cfg.parts # 10
        self.map_size = 7 # Z是2048*7*7
        # TODO 这里是否
        self.pool = nn.MaxPool2d(self.map_size, self.map_size)
        self.cov = nn.Conv2d(self.cov_channel, self.parts, 1) # 2048->10
        self.p_linear = nn.Linear(self.cov_channel*self.parts, d, False) # 2048*10 -> embedding dimension
        self.dropout2 = nn.Dropout(0.4)

        self.conv_bilinear = nn.Linear(self.cov_channel, self.compress_map, 1)
        self.b_linear = nn.Linear(self.compress_map * self.cov_channel, d, False)
        self.dropout3 = nn.Dropout(0.4)
        # TODO 先冻结参数
        # for param in self.features.parameters():
        #     param.requires_grad = False
    def forward(self, x):
        features = self.features(x) # [20/22, 2048, 7, 7]
        w = features.size()
        # MaskGenerate # weighs.size: [20/22, 10, 7, 7]
        weights = torch.sigmoid(self.cov(features)) # # batch x parts x 7 x 7
        batch,parts,width,height = weights.size()
        weights_layout = weights.view(batch,-1) # [bs, 490]
        # 找个全局最大值
        threshold_value,_ = weights_layout.max(dim=1) # [bs]
        # 找到每个Mask里面最大的
        local_max,_ = weights.view(batch,parts,-1).max(dim=2) # [bs, 10]
        # map_threshold为适应性系数
        threshold_value = self.map_threshold*threshold_value.view(batch,1) \
            .expand(batch,parts) # [bs, 10]
        # 下面这句话之后即得到经过AT之后的
        weights = weights*local_max.ge(threshold_value).view(batch,parts,1,1). \
            float().expand(batch,parts,width,height) # [bs, 10, 7, 7]

        blocks = []
        for k in range(self.parts):
            # 即[20, 2048, 7, 7] * [20, 1, 7, 7].reshape(20, 2048, 7, 7)
            Y = features*weights[:,k,:,:]. \
                unsqueeze(dim=1). \
                expand(w[0],self.cov_channel,w[2],w[3]) # [20, 2048, 7, 7]
            blocks.append(self.pool(Y).squeeze().view(-1,self.cov_channel)) # 元素为[20/22, 2048]
        # 最后这10个特征图融合了来作为最后的特征
        p_output = self.dropout2(self.p_linear(torch.cat(blocks, dim=1))) # torch.cat()后:[20/22, 20480]
        
        if self.training:
            p_out = self.train_linear(p_output)
        else:
            p_out = self.test_linear(p_output)
        
        return p_out

class GZSL_Multi_HFLN(nn.Module):
    '''
        多分支
    '''
    def __init__(self, cfg, model, train_attr=None, test_attr=None, father_train_attr=None, father_test_attr=None):
        super(GZSL_Multi_HFLN, self).__init__()
        print('GZSL_Multi_HFLN:train_attr', train_attr)
        train_attr = Parameter(torch.from_numpy(train_attr).float().cuda())
        test_attr = Parameter(torch.from_numpy(test_attr).float().cuda())
        father_train_attr = Parameter(torch.from_numpy(father_train_attr).float().cuda())
        father_test_attr = Parameter(torch.from_numpy(father_test_attr).float().cuda())
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.father_features = self.features[0:6]

        self.cov_channel = 2048
        self.map_size = 7
        '''
            2021-05-12输入尺寸为448的图像 对father分支和主分支的的map_size都要修改
        '''
        # self.map_size = 14
        self.compress_map = 20

        self.father_cov_channel = 512
        self.father_map_size = 28
        # self.father_map_size = 56 # 28->56
        
        f_c1, d = father_train_attr.size()
        f_c2, _ = father_test_attr.size()
        self.father_train_linear = nn.Linear(d, f_c1, False)
        self.father_train_linear.weight = father_train_attr
        for para in self.father_train_linear.parameters():
            para.requires_grad = False
        self.father_test_linear = nn.Linear(d, f_c2, False)
        self.father_test_linear.weight = father_test_attr
        for para in self.father_test_linear.parameters():
            para.requires_grad = False

        c1, d = train_attr.size()
        c2, _ = test_attr.size()
        # c3, _ = val_attr.size()
        self.train_linear = nn.Linear(d, c1, False)
        self.train_linear.weight = train_attr
        # 固定这些权重
        for para in self.train_linear.parameters():
            para.requires_grad = False
        # self.val_linear = nn.Linear(d, c3, False)
        # self.val_linear.weight = val_attr
        # for para in self.val_linear.parameters():
        #     para.requires_grad = False
        self.test_linear = nn.Linear(d, c2, False)
        self.test_linear.weight = test_attr
        for para in self.test_linear.parameters():
            para.requires_grad = False

        # aren
        self.map_threshold = cfg.threshold # 0.8
        self.parts = cfg.parts # 10
        self.map_size = 7 # Z是2048*7*7
        '''
            2021-05-01修改输入图像为448，则此处map_size变为14
        '''
        # self.map_size = 14 # Z是2048*14*14
        # TODO 这里是否
        self.pool = nn.MaxPool2d(self.map_size, self.map_size)
        self.cov = nn.Conv2d(self.cov_channel, self.parts, 1) # 2048->10
        self.p_linear = nn.Linear(self.cov_channel*self.parts, d, False) # 2048*10 -> embedding dimension
        self.dropout2 = nn.Dropout(0.4)

        self.father_pool = nn.MaxPool2d(self.father_map_size, self.father_map_size)
        self.father_cov = nn.Conv2d(self.father_cov_channel, self.parts, 1) # 2048->10
        self.father_p_linear = nn.Linear(self.father_cov_channel*self.parts, d, False) # 2048*10 -> embedding dimension
        self.father_dropout2 = nn.Dropout(0.4)

        self.conv_bilinear = nn.Linear(self.cov_channel, self.compress_map, 1)
        self.b_linear = nn.Linear(self.compress_map * self.cov_channel, d, False)
        self.dropout3 = nn.Dropout(0.4)
        # TODO 先冻结参数
        # for param in self.features.parameters():
        #     param.requires_grad = False
    def forward(self, x):
        father_features = self.father_features(x)
        w = father_features.size()
        weights = torch.sigmoid(self.father_cov(father_features))
        batch, parts, width, height = weights.size()
        weights_layout = weights.view(batch, -1)
        threshold_value,_ = weights_layout.max(dim=1) # [bs]
        # 找到每个Mask里面最大的
        local_max,_ = weights.view(batch,parts,-1).max(dim=2) # [bs, 10]
        # map_threshold为适应性系数
        threshold_value = self.map_threshold*threshold_value.view(batch,1) \
            .expand(batch,parts) # [bs, 10]
        # 下面这句话之后即得到经过AT之后的
        weights = weights*local_max.ge(threshold_value).view(batch,parts,1,1). \
            float().expand(batch,parts,width,height) # [bs, 10, 7, 7]
        blocks = []
        for k in range(self.parts):
            # 即[20, 2048, 7, 7] * [20, 1, 7, 7].reshape(20, 2048, 7, 7)
            Y = father_features*weights[:,k,:,:]. \
                unsqueeze(dim=1). \
                expand(w[0],self.father_cov_channel,w[2],w[3]) # [20, 2048, 7, 7]
            blocks.append(self.father_pool(Y).squeeze().view(-1,self.father_cov_channel)) # 元素为[20/22, 2048]
        # 最后这10个特征图融合了来作为最后的特征
        father_p_output = self.father_dropout2(self.father_p_linear(torch.cat(blocks, dim=1))) # torch.cat()后:[20/22, 20480]
        if self.training:
            father_p_out = self.father_train_linear(father_p_output)
        else:
            father_p_out = self.father_test_linear(father_p_output)
        
        features = self.features(x) # [20/22, 2048, 7, 7]
        w = features.size()
        # MaskGenerate # weighs.size: [20/22, 10, 7, 7]
        weights = torch.sigmoid(self.cov(features)) # # batch x parts x 7 x 7
        batch,parts,width,height = weights.size()
        weights_layout = weights.view(batch,-1) # [bs, 490]
        # 找个全局最大值
        threshold_value,_ = weights_layout.max(dim=1) # [bs]
        # 找到每个Mask里面最大的
        local_max,_ = weights.view(batch,parts,-1).max(dim=2) # [bs, 10]
        # map_threshold为适应性系数
        threshold_value = self.map_threshold*threshold_value.view(batch,1) \
            .expand(batch,parts) # [bs, 10]
        # 下面这句话之后即得到经过AT之后的
        weights = weights*local_max.ge(threshold_value).view(batch,parts,1,1). \
            float().expand(batch,parts,width,height) # [bs, 10, 7, 7]

        blocks = []
        for k in range(self.parts):
            # 即[20, 2048, 7, 7] * [20, 1, 7, 7].reshape(20, 2048, 7, 7)
            Y = features*weights[:,k,:,:]. \
                unsqueeze(dim=1). \
                expand(w[0],self.cov_channel,w[2],w[3]) # [20, 2048, 7, 7]
            blocks.append(self.pool(Y).squeeze().view(-1,self.cov_channel)) # 元素为[20/22, 2048]
        # 最后这10个特征图融合了来作为最后的特征
        p_output = self.dropout2(self.p_linear(torch.cat(blocks, dim=1))) # torch.cat()后:[20/22, 20480]
        
        if self.training:
            p_out = self.train_linear(p_output)
        else:
            p_out = self.test_linear(p_output)
            
        return p_out, father_p_out
class GZSL_Multi_HFLN_Father(nn.Module):
    '''
        GZSL设置下，多分支，仅包含上位分支
        用于先训练上位分支，然后嵌入到主分支
    '''
    def __init__(self, cfg, model, father_train_attr=None, father_test_attr=None):
        super(GZSL_Multi_HFLN_Father, self).__init__()
        father_train_attr = Parameter(torch.from_numpy(father_train_attr).float().cuda())
        father_test_attr = Parameter(torch.from_numpy(father_test_attr).float().cuda())
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.father_features = self.features[0:6]

        self.cov_channel = 2048
        self.map_size = 7
        self.compress_map = 20

        self.father_cov_channel = 512
        self.father_map_size = 28
        
        f_c1, d = father_train_attr.size()
        f_c2, _ = father_test_attr.size()
        self.father_train_linear = nn.Linear(d, f_c1, False)
        self.father_train_linear.weight = father_train_attr
        for para in self.father_train_linear.parameters():
            para.requires_grad = False
        self.father_test_linear = nn.Linear(d, f_c2, False)
        self.father_test_linear.weight = father_test_attr
        for para in self.father_test_linear.parameters():
            para.requires_grad = False

        # aren
        self.map_threshold = cfg.threshold # 0.8
        self.parts = cfg.parts # 10
        self.map_size = 7 # Z是2048*7*7
        # TODO 这里是否

        self.father_pool = nn.MaxPool2d(self.father_map_size, self.father_map_size)
        self.father_cov = nn.Conv2d(self.father_cov_channel, self.parts, 1) # 2048->10
        self.father_p_linear = nn.Linear(self.father_cov_channel*self.parts, d, False) # 2048*10 -> embedding dimension
        self.father_dropout2 = nn.Dropout(0.4)

        self.conv_bilinear = nn.Linear(self.cov_channel, self.compress_map, 1)
        self.b_linear = nn.Linear(self.compress_map * self.cov_channel, d, False)
        self.dropout3 = nn.Dropout(0.4)
        # TODO 先冻结参数
        # for param in self.features.parameters():
        #     param.requires_grad = False
    def forward(self, x):
        father_features = self.father_features(x)
        w = father_features.size()
        weights = torch.sigmoid(self.father_cov(father_features))
        batch, parts, width, height = weights.size()
        weights_layout = weights.view(batch, -1)
        threshold_value,_ = weights_layout.max(dim=1) # [bs]
        # 找到每个Mask里面最大的
        local_max,_ = weights.view(batch,parts,-1).max(dim=2) # [bs, 10]
        # map_threshold为适应性系数
        threshold_value = self.map_threshold*threshold_value.view(batch,1) \
            .expand(batch,parts) # [bs, 10]
        # 下面这句话之后即得到经过AT之后的
        weights = weights*local_max.ge(threshold_value).view(batch,parts,1,1). \
            float().expand(batch,parts,width,height) # [bs, 10, 7, 7]
        blocks = []
        for k in range(self.parts):
            # 即[20, 2048, 7, 7] * [20, 1, 7, 7].reshape(20, 2048, 7, 7)
            Y = father_features*weights[:,k,:,:]. \
                unsqueeze(dim=1). \
                expand(w[0],self.father_cov_channel,w[2],w[3]) # [20, 2048, 7, 7]
            blocks.append(self.father_pool(Y).squeeze().view(-1,self.father_cov_channel)) # 元素为[20/22, 2048]
        # 最后这10个特征图融合了来作为最后的特征
        father_p_output = self.father_dropout2(self.father_p_linear(torch.cat(blocks, dim=1))) # torch.cat()后:[20/22, 20480]
        if self.training:
            father_p_out = self.father_train_linear(father_p_output)
        else:
            father_p_out = self.father_test_linear(father_p_output)
            
        return father_p_out
class GZSL_Multi_Combine_HFLN(nn.Module):
    '''
        多分支
    '''
    def __init__(self, cfg, model, train_attr=None, test_attr=None, father_train_attr=None, father_test_attr=None):
        super(GZSL_Multi_Combine_HFLN, self).__init__()
        train_attr = Parameter(torch.from_numpy(train_attr).float().cuda())
        test_attr = Parameter(torch.from_numpy(test_attr).float().cuda())
        father_train_attr = Parameter(torch.from_numpy(father_train_attr).float().cuda())
        father_test_attr = Parameter(torch.from_numpy(father_test_attr).float().cuda())
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.father_features = self.features[0:6]

        self.cov_channel = 2048
        self.map_size = 7
        self.compress_map = 20

        self.father_cov_channel = 512
        self.father_map_size = 28
        
        f_c1, d = father_train_attr.size()
        f_c2, _ = father_test_attr.size()
        self.father_train_linear = nn.Linear(d, f_c1, False)
        self.father_train_linear.weight = father_train_attr
        for para in self.father_train_linear.parameters():
            para.requires_grad = False
        self.father_test_linear = nn.Linear(d, f_c2, False)
        self.father_test_linear.weight = father_test_attr
        for para in self.father_test_linear.parameters():
            para.requires_grad = False

        c1, d = train_attr.size()
        c2, _ = test_attr.size()
        # c3, _ = val_attr.size()
        self.train_linear = nn.Linear(d, c1, False)
        self.train_linear.weight = train_attr
        # 固定这些权重
        for para in self.train_linear.parameters():
            para.requires_grad = False
        # self.val_linear = nn.Linear(d, c3, False)
        # self.val_linear.weight = val_attr
        # for para in self.val_linear.parameters():
        #     para.requires_grad = False
        self.test_linear = nn.Linear(d, c2, False)
        self.test_linear.weight = test_attr
        for para in self.test_linear.parameters():
            para.requires_grad = False

        # aren
        self.map_threshold = cfg.threshold # 0.8
        self.parts = cfg.parts # 10
        self.map_size = 7 # Z是2048*7*7
        # TODO 这里是否
        self.pool = nn.MaxPool2d(self.map_size, self.map_size)
        self.cov = nn.Conv2d(self.cov_channel, self.parts, 1) # 2048->10
        self.p_linear = nn.Linear(self.cov_channel*self.parts, d, False) # 2048*10 -> embedding dimension
        self.dropout2 = nn.Dropout(0.4)

        self.father_pool = nn.MaxPool2d(self.father_map_size, self.father_map_size)
        self.father_cov = nn.Conv2d(self.father_cov_channel, self.parts, 1) # 2048->10
        self.father_p_linear = nn.Linear(self.father_cov_channel*self.parts, d, False) # 2048*10 -> embedding dimension
        self.father_dropout2 = nn.Dropout(0.4)

        self.conv_bilinear = nn.Linear(self.cov_channel, self.compress_map, 1)
        self.b_linear = nn.Linear(self.compress_map * self.cov_channel, d, False)
        self.dropout3 = nn.Dropout(0.4)
        # TODO 先冻结参数
        # for param in self.features.parameters():
        #     param.requires_grad = False
    def forward(self, x):
        father_features = self.father_features(x)
        w = father_features.size()
        weights = torch.sigmoid(self.father_cov(father_features))
        batch, parts, width, height = weights.size()
        weights_layout = weights.view(batch, -1)
        threshold_value,_ = weights_layout.max(dim=1) # [bs]
        # 找到每个Mask里面最大的
        local_max,_ = weights.view(batch,parts,-1).max(dim=2) # [bs, 10]
        # map_threshold为适应性系数
        threshold_value = self.map_threshold*threshold_value.view(batch,1) \
            .expand(batch,parts) # [bs, 10]
        # 下面这句话之后即得到经过AT之后的
        weights = weights*local_max.ge(threshold_value).view(batch,parts,1,1). \
            float().expand(batch,parts,width,height) # [bs, 10, 7, 7]
        blocks = []
        for k in range(self.parts):
            # 即[20, 2048, 7, 7] * [20, 1, 7, 7].reshape(20, 2048, 7, 7)
            Y = father_features*weights[:,k,:,:]. \
                unsqueeze(dim=1). \
                expand(w[0],self.father_cov_channel,w[2],w[3]) # [20, 2048, 7, 7]
            blocks.append(self.father_pool(Y).squeeze().view(-1,self.father_cov_channel)) # 元素为[20/22, 2048]
        # 最后这10个特征图融合了来作为最后的特征
        father_p_output = self.father_dropout2(self.father_p_linear(torch.cat(blocks, dim=1))) # torch.cat()后:[20/22, 20480]
        # if self.training:
        #     father_p_out = self.father_train_linear(father_p_output)
        # else:
        #     father_p_out = self.father_test_linear(father_p_output)
        
        features = self.features(x) # [20/22, 2048, 7, 7]
        w = features.size()
        # MaskGenerate # weighs.size: [20/22, 10, 7, 7]
        weights = torch.sigmoid(self.cov(features)) # # batch x parts x 7 x 7
        batch,parts,width,height = weights.size()
        weights_layout = weights.view(batch,-1) # [bs, 490]
        # 找个全局最大值
        threshold_value,_ = weights_layout.max(dim=1) # [bs]
        # 找到每个Mask里面最大的
        local_max,_ = weights.view(batch,parts,-1).max(dim=2) # [bs, 10]
        # map_threshold为适应性系数
        threshold_value = self.map_threshold*threshold_value.view(batch,1) \
            .expand(batch,parts) # [bs, 10]
        # 下面这句话之后即得到经过AT之后的
        weights = weights*local_max.ge(threshold_value).view(batch,parts,1,1). \
            float().expand(batch,parts,width,height) # [bs, 10, 7, 7]

        blocks = []
        for k in range(self.parts):
            # 即[20, 2048, 7, 7] * [20, 1, 7, 7].reshape(20, 2048, 7, 7)
            Y = features*weights[:,k,:,:]. \
                unsqueeze(dim=1). \
                expand(w[0],self.cov_channel,w[2],w[3]) # [20, 2048, 7, 7]
            blocks.append(self.pool(Y).squeeze().view(-1,self.cov_channel)) # 元素为[20/22, 2048]
        # 最后这10个特征图融合了来作为最后的特征
        p_output = self.dropout2(self.p_linear(torch.cat(blocks, dim=1))) # torch.cat()后:[20/22, 20480]
        
        if self.training:
            p_out = self.train_linear(p_output+father_p_output)
        else:
            p_out = self.test_linear(p_output+father_p_output)
        
        return p_out
class HFLN(nn.Module):
    def __init__(self, cfg, model, train_attr=None, test_attr=None):
        super(HFLN, self).__init__()
        # self.features = nn.Sequential(*list(model.children())[:-2])
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(2048, cfg.dimension)
        # TODO 先冻结参数
        for param in self.features.parameters():
            param.requires_grad = False
    def forward(self, x):
        x = self.features(x).squeeze() # [64, 2048, 1, 1] -> squeeze
        # x = F.normalize(x, dim=1)
        # TODO 提取后的特征没做归一化
        x = self.fc(x)
        return x
