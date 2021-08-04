from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
class Dataset(data.Dataset):
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
