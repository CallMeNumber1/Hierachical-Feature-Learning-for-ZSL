from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
class Multi_Dataset(data.Dataset):
    '''
        用于多分支的数据集，包括动物和上位类别的标签
    '''
    def __init__(self, args, examples, labels, embeddings, transfrom, father_labels):
        # 初始化
        self.examples = examples
        self.labels = labels
        self.embeddings = embeddings
        self.transform = transfrom
        self.image_dir = args.image_dir
        self.args = args
        self.father_labels = father_labels
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        filename = self.examples[index]
        X = Image.open(self.image_dir+filename).convert('RGB')
        X = self.transform(X)
        label = self.labels[index]
        embd = self.embeddings[index]
        father_label = self.father_labels[index]
        return X, label, embd, father_label
class Dataset(data.Dataset):
    '''
        用于单分支的数据集，只有动物的标签
    '''
    def __init__(self, args, examples, labels, embeddings, transfrom):
        # 初始化
        self.examples = examples
        self.labels = labels
        self.embeddings = embeddings
        self.transform = transfrom
        self.image_dir = args.image_dir
        self.args = args
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        filename = self.examples[index]
        X = Image.open(self.image_dir+filename).convert('RGB')
        X = self.transform(X)
        label = self.labels[index]
        embd = self.embeddings[index]
        return X, label, embd
