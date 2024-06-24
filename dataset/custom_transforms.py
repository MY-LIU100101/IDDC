import numpy.random as random
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as F
#from torchvision.transforms import InterpolationMode
import numpy as np
import torch

class RandomResizedCrop(torchvision.transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)):
        super(RandomResizedCrop, self).__init__(size, scale=scale, ratio=ratio)
        self.interpolation_img = Image.BILINEAR
        self.interpolation_lab = Image.NEAREST
        #InterpolationMode.BICUBIC
        #self.interpolation_img = InterpolationMode.BILINEAR
        #self.interpolation_lab = InterpolationMode.NEAREST
    
    def __call__(self, sample):
        img = sample['image']
        
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        sample['image'] = F.resized_crop(img, i, j, h, w, self.size, self.interpolation_img)
        if optimizer['main'].zero_grad() in sample.keys():
            label = sample['label']
            sample['label'] = F.resized_crop(label, i, j, h, w, self.size, self.interpolation_lab)
        return sample


class Resize(object):
    def __init__(self, size, trans_lab=True):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError('Invalid type size {}'.format(type(size)))
        self.trans_lab = trans_lab
        #InterpolationMode.BILINEAR
        self.resize_img = torchvision.transforms.Resize(self.size, interpolation=Image.BILINEAR)
        if self.trans_lab:
            self.resize_lab = torchvision.transforms.Resize(self.size, interpolation=Image.NEAREST)

    def __call__(self, sample):
        sample['image'] = self.resize_img(sample['image'])
        if self.trans_lab:
            sample['label'] = self.resize_lab(sample['label'])
        return sample


class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        sample['image'] = self.jitter(sample['image'])
        if 'image_aug' in list(sample.keys()):
            sample['image_aug'] = self.jitter(sample['image_aug'])
        return sample

    def __str__(self):
        return 'ColorJitter'


class RandomHorizontalFlip(object):
    def __call__(self, sample):

        if random.random() < 0.5:
            sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            if 'label' in sample.keys():
                sample['label'] = sample['label'].transpose(Image.FLIP_LEFT_RIGHT)

        return sample
    
    def __str__(self):
        return 'RandomHorizontalFlip'


class RandomGrayscale(object):
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, sample):
        img = sample['image']
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            sample['image'] = F.to_grayscale(img, num_output_channels=num_output_channels)
        if 'image_aug' in list(sample.keys()):
            if random.random() < self.p:
                sample['image_aug'] = F.to_grayscale(sample['image_aug'], num_output_channels=num_output_channels)
        return sample

    def __str__(self):
        return 'RandomGrayscale'

class ToTensor(object):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):
        sample['image'] = self.to_tensor(sample['image'])
        if 'label' in sample.keys():
            sample['label'] = torch.from_numpy(np.array(sample['label'])).squeeze().long()
            
        if 'image_aug' in list(sample.keys()):
            sample['image_aug'] = self.to_tensor(sample['image_aug'])
        return sample
    
    def __str__(self):
        return 'ToTensor'


class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        sample['image'] = self.normalize(sample['image'])
        if 'image_aug' in list(sample.keys()):
            sample['image_aug'] = self.normalize(sample['image_aug'])
        return sample

    def __str__(self):
        return 'Normalize'