from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import numpy as np
import albumentations as albu

# 数据增强
def get_transforms(size):
    aug_fn = albu.Compose([albu.HorizontalFlip(),  # 水平翻转
                           albu.VerticalFlip(),  # 垂直翻转
                           albu.RandomRotate90()  # 随机旋转90度
                           ])
    crop_fn = albu.RandomCrop(size, size, always_apply=True) # 裁剪输入的随机部分
    pipeline = albu.Compose([aug_fn, crop_fn], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process

# 数据标准化
def get_normalize():
    transform = transforms.ToTensor()

    def process(a, b):
        image = transform(a).permute(1, 2, 0) - 0.5
        target = transform(b).permute(1, 2, 0) - 0.5
        return image, target

    return process

# 打包sharp和blur
class PairedDataset(Dataset):
    def __init__(self, files_a, files_b, transform_fn, normalize_fn):
        """files_a:blur, files_b:sharp"""
        self.data_a = files_a
        self.data_b = files_b
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        
    def preprocess(self, img, res):
        def transpose(x):
            return np.transpose(x, (2, 0, 1))

        return map(transpose, self.normalize_fn(img, res))
    
    def __len__(self):
        return len(self.data_a)
        
    def __getitem__(self, idx):
        a, b = self.data_a[idx], self.data_b[idx]
        a = cv2.imread(a)
        b = cv2.imread(b)
        a, b = self.transform_fn(a, b)
        a, b = self.preprocess(a, b)
        return {'a': a, 'b': b}
