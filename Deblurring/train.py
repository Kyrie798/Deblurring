import os
from metric_counter import MetricCounter
from models.losses import Stripformer_Loss
from models.models import DeblurModel
from models.Stripformer import Stripformer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from torch.utils.data import DataLoader
import tqdm
import torch
from joblib import cpu_count
from dataset import PairedDataset
from dataset import get_transforms
from dataset import get_normalize
from glob import glob

num_epochs = 3000
class Trainer:
    def __init__(self, train, val):
        self.train_dataset = train
        self.val_dataset = val
        self.metric_counter = MetricCounter('Stripformer_pretrained')

    def train(self):
        self.criterionG = Stripformer_Loss()
        self.netG = Stripformer()
        self.netG.cuda()
        self.model = DeblurModel()
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=0.0001)
        self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=num_epochs, eta_min=0.0000001)
        start_epoch = 0
        if os.path.exists('last_Stripformer_pretrained.pth'):
            print('load_pretrained')
            training_state = (torch.load('last_Stripformer_pretrained.pth'))
            start_epoch = training_state['epoch']
            new_weight = self.netG.state_dict()
            new_weight.update(training_state['model_state'])
            self.netG.load_state_dict(new_weight)
            new_optimizer = self.optimizer_G.state_dict()
            new_optimizer.update(training_state['optimizer_state'])
            self.optimizer_G.load_state_dict(new_optimizer)
            new_scheduler = self.scheduler_G.state_dict()
            new_scheduler.update(training_state['scheduler_state'])
            self.scheduler_G.load_state_dict(new_scheduler)
        
        for epoch in range(start_epoch, num_epochs-1):
            self.run_epoch(epoch)
            if epoch % 30 == 0 or epoch == (num_epochs-1):
                self.validate(epoch)
            self.scheduler_G.step()
            scheduler_state = self.scheduler_G.state_dict()
            training_state = {'epoch': epoch,  'model_state': self.netG.state_dict(),
                              'scheduler_state': scheduler_state, 'optimizer_state': self.optimizer_G.state_dict()}
            if self.metric_counter.update_best_model():
                torch.save(training_state['model_state'], 'best_{}.pth'.format('Stripformer_pretrained'))
            if epoch % 300 == 0:
                torch.save(training_state, 'last_{}_{}.pth'.format('Stripformer_pretrained', epoch))
            if epoch == (num_epochs-1):
                torch.save(training_state['model_state'], 'final_{}.pth'.format('Stripformer_pretrained'))
            torch.save(training_state, 'last_{}.pth'.format('Stripformer_pretrained'))
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                'Stripformer_pretrained', epoch, self.metric_counter.loss_message()))
        
    def run_epoch(self, epoch):
        self.metric_counter.clear()
        # 把训练参数分别赋予不同的学习率
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']
        epoch_size = len(self.train_dataset)
        # 进度条
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            inputs, targets = self.model.get_input(data)
            outputs = self.netG(inputs)
            self.optimizer_G.zero_grad()
            loss_G = self.criterionG(outputs, targets, inputs)
            loss_G.backward()
            self.optimizer_G.step()
            self.metric_counter.add_losses(loss_G.item())
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            # 设置进度条显示信息
            tq.set_postfix(loss=self.metric_counter.loss_message())
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)
        
    def validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            with torch.no_grad():
                inputs, targets = self.model.get_input(data)
                outputs = self.netG(inputs)
                loss_G = self.criterionG(outputs, targets, inputs)
                self.metric_counter.add_losses(loss_G.item())
                curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
                self.metric_counter.add_metrics(curr_psnr, curr_ssim)
                if not i:
                    self.metric_counter.add_image(img_for_vis, tag='val')
                i += 1
                if i > epoch_size:
                    break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)

if __name__ == '__main__':
    transform_fn = get_transforms(256)
    normalize_fn = get_normalize()

    train_blur = sorted(glob("./datasets/GoPro/train/blur/**/*.png", recursive=True))
    train_sharp = sorted(glob("./datasets/GoPro/train/sharp/**/*.png", recursive=True))
    val_blur = sorted(glob("./datasets/GoPro/test/blur/**/*.png", recursive=True))
    val_sharp = sorted(glob("./datasets/GoPro/test/blur/**/*.png", recursive=True))

    train_datasets = PairedDataset(train_blur, train_sharp, transform_fn, normalize_fn)
    val_datasets = PairedDataset(val_blur, val_sharp, transform_fn, normalize_fn)

    train = DataLoader(train_datasets, batch_size=2, num_workers=cpu_count(), shuffle=True)
    val = DataLoader(val_datasets, batch_size=2, num_workers=cpu_count(), shuffle=True)
    trainer = Trainer(train=train, val=val)
    trainer.train()