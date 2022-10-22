from glob import glob
from torch.utils.data import Dataset
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms

# 数据增强
get_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomCrop(256),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(90)])

train_blur = sorted(glob("./datasets/GoPro/train/blur/**/*.png", recursive=True))
train_sharp = sorted(glob("./datasets/GoPro/train/sharp/**/*.png", recursive=True))
val_blur = sorted(glob("./datasets/GoPro/test/blur/**/*.png", recursive=True))
val_sharp = sorted(glob("./datasets/GoPro/test/blur/**/*.png", recursive=True))

# 打包sharp和blur
class PairedDataset(Dataset):
    def __init__(self, files_a, files_b, transforms):
        """files_a:blur, files_b:sharp"""
        self.data_a = files_a
        self.data_b = files_b
        self.transforms = transforms
    def __len__(self):
        return len(self.data_a)
    def __getitem__(self, idx):
        a, b = self.data_a[idx], self.data_b[idx]
        a = cv2.imread(a)
        b = cv2.imread(b)
        a = self.transforms(a)
        b = self.transforms(b)
        return {'a': a, 'b': b}

train_datasets = PairedDataset(train_blur, train_sharp, get_transforms)
val_datasets = PairedDataset(val_blur, val_sharp, get_transforms)

train = DataLoader(train_datasets, batch_size=8, num_workers=0, shuffle=True)
val = DataLoader(val_datasets, batch_size=8, num_workers=0, shuffle=True)

# 验证过程
print(len(train_datasets))
print(len(val_datasets))

print(len(train))
print(len(val))

print(train_datasets.__getitem__(0)['a'].shape)
print(train_datasets.__getitem__(0)['b'].shape)
print(val_datasets.__getitem__(0)['a'].shape)
print(val_datasets.__getitem__(0)['b'].shape)

train_test = next(iter(train))
print(train_test['a'].shape)

