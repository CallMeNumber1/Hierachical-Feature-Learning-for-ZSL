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
class NEW_HFLN(nn.Module):
    def __init__(self, cfg, model, train_attr=None, test_attr=None):
        super(NEW_HFLN, self).__init__()
        train_attr = Parameter(torch.from_numpy(train_attr).float().cuda())
        test_attr = Parameter(torch.from_numpy(test_attr).float().cuda())
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
