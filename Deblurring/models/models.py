import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as SSIM

from util.metrics import PSNR


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        inputs = data['a']
        targets = data['b']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0
        return image_numpy

    def get_images_and_metrics(self, inp, output, target):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake.astype('uint8'), real.astype('uint8'), multichannel=True)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img
