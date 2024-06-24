import os
import torch 
import torch.nn as nn
import math
import torchvision.transforms.functional as F
import torch.nn.functional as F_func
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import dataset.custom_transforms as cus_transforms
import cv2
import copy
import random
import numpy as np
from skimage.segmentation import felzenszwalb, mark_boundaries

from skimage import io


class CocoStuff27(data.Dataset):
    def __init__(self, path_dataset, split='train', val_mode=0):
        super(CocoStuff27, self).__init__()

        self.split = split
        assert self.split in ['train', 'val']
        self.path_dataset = path_dataset
        self.val_mode = val_mode

        self.file_train_subject, self.file_val_subject = \
                self.check_datapath_integrity(self.path_dataset)

        self.imdb = self.load_imdb()
        self.label_mapper = self.get_mapper()
        self.train_geo_transformations, self.train_color_transformations, \
        self.train_plain_transformations, self.train_weak_color_transformations = \
                self.get_train_transofrmations()
        self.val_transformations = self.get_val_transofrmations()
        self.geo_transformations = self.get_geo_transformations()

        self.val_resize_trans = transforms.Compose([
                        cus_transforms.Resize(size=320),
                        cus_transforms.ToTensor(),
                        cus_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.image_files = [os.path.join(self.path_dataset, 'images', split+'2017', i+'.jpg') for i in self.imdb]
        self.label_files = [os.path.join(self.path_dataset, 'annotations', split+'2017', i+'.png') for i in self.imdb]

    def __getitem__(self, index):
        if self.split == 'train':
            img = cv2.imread(self.image_files[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lab = cv2.imread(self.label_files[index], cv2.IMREAD_GRAYSCALE)
        

            img, lab = self.random_horizental_flip(img, lab)

            img = np.array(img)
            lab = self.label_mapper(lab)

            img, lab = self.random_resize(img, lab, [0.8, 1.2], [0.8, 1.2])
            img_ori, img_aug, lab_ori, lab_aug, _, _ = self.random_crop_overlap(img, lab, crop_size = 224)
           
            lab_ori = Image.fromarray(lab_ori.astype('uint8'))
            img_ori = Image.fromarray(img_ori.astype('uint8'))
            img_aug = Image.fromarray(img_aug.astype('uint8'))
            
            sample = self.train_plain_transformations({'image': img_ori, 'label':lab_ori})
            sample_aug = self.train_plain_transformations({'image': img_aug})

            sample_color_aug = self.train_color_transformations({'image': img_ori})
            sample['image_color_aug'] = sample_color_aug['image']
            sample['image_aug'] = sample_aug['image']
        
        else:
            img = Image.open(self.image_files[index]).convert('RGB')
            lab = Image.open(self.label_files[index]).convert('L')

            img_name = self.image_files[index]

            lab = np.array(lab)
            ori_shape = np.shape(lab)
            lab_mapped = self.label_mapper(lab)
            lab = Image.fromarray(lab_mapped.astype('uint8'))

            ####################################
            '''
            _w, _h = lab.size
            if _w <= _h:
                cr_pos = (0, (_h-_w)//2,_w ,(_h-_w)//2+_w) 
            elif _w>_h:
                cr_pos = ((_w-_h)//2, 0, (_w-_h)//2+_h,_h) 
            else:
                raise NotImplementedError
            img = img.crop(cr_pos)
            lab = lab.crop(cr_pos)
            ori_shape = (320, 320)
            '''
            #####################################

            _w, _h = lab.size
            if _w <= _h and _w < 224:
                img = img.resize((224, np.ceil(_h/_w*224)), resample=Image.BILINEAR)
                lab = lab.resize((224, np.ceil(_h/_w*224)), resample=Image.NEAREST)
            elif _h < _w and _h < 224:
                img = img.resize((np.ceil(_w/_h*224), 224), resample=Image.BILINEAR)
                lab = lab.resize((np.ceil(_w/_h*224), 224), resample=Image.NEAREST)
                
            if self.val_mode==0:
                sample = self.val_transformations({'image': img, 'label': lab, 'shape': ori_shape})
                patches, poses = self.slidding_val(sample['image'], crop=224, overlap=75)
                sample['patches'] = patches
                sample['poses'] = poses
                sample['img_name'] = img_name

            elif self.val_mode==1:
                
                sample = self.val_resize_trans({'image': img, 'label': lab, 'shape': ori_shape})
                sample['label_ori'] = np.array(lab)

            elif self.val_mode==2:
                sample = self.val_resize_trans({'image': img, 'label': lab, 'shape': ori_shape})

        return sample

    def get_val_transofrmations(self):
        return transforms.Compose([
            cus_transforms.ToTensor(),
            cus_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    def slidding_val(self, img, crop=224, overlap=75):
        h, w = img.shape[-2:]
        assert h>=crop and w>=crop, '{}, {}'.format(h, w)

        stride = crop-overlap

        num_h = math.ceil((h - crop) / stride + 1)
        num_w = math.ceil((w - crop) / stride + 1)

        all_patches = torch.zeros(num_h*num_w, 3, crop, crop)
        all_poses = []

        for _h in range(num_h):
            for _w in range(num_w):
                h0 = min(_h*stride, h-crop)
                h1 = min(_h*stride+crop, h)

                w0 = min(_w*stride, w-crop)
                w1 = min(_w*stride+crop, w)

                all_patches[_h*num_w+_w, :,:,:] = img[:, h0:h1, w0:w1]
                all_poses.append((h0, h1, w0, w1))

        return all_patches, all_poses


    
    def get_train_transofrmations(self):
        geo_transform =  transforms.Compose([
            cus_transforms.RandomHorizontalFlip(),
            cus_transforms.RandomResizedCrop(size=224, scale=[0.8, 1.2], ratio=[3/4., 4/3.]),])

        color_transform =  transforms.Compose([
            #transforms.RandomApply([
            cus_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            #], p=0.6),
            cus_transforms.RandomGrayscale(p=0.1),
            cus_transforms.ToTensor(),
            cus_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        plain_transform =  transforms.Compose([
            cus_transforms.ToTensor(),
            cus_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        weak_color_transform =  transforms.Compose([
            transforms.RandomApply([
                cus_transforms.ColorJitter(0.1, 0.1, 0.1, 0.025),
            ], p=0.5),
            #cus_transforms.RandomGrayscale(p=0.1),
            cus_transforms.ToTensor(),
            cus_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        return geo_transform, color_transform, plain_transform, weak_color_transform 

    

    def get_geo_transformations(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(224,224), scale=(0.8, 1.0))
            ])

    def random_crop_overlap(self, img, lab, crop_size = 224):
        w, h = img.shape[1], img.shape[0]
        p = np.random.rand(4) # [w1, w2, h1, h2]
        c0_w = round((w - crop_size) * p[0])
        c0_h = round((h - crop_size) * p[1])

        c1_w_range = (max(c0_w-crop_size//2, 0), min(c0_w+crop_size//2, w-crop_size))
        c1_h_range = (max(c0_h-crop_size//2, 0), min(c0_h+crop_size//2, h-crop_size))
        c1_w = c1_w_range[0] + round((c1_w_range[1] - c1_w_range[0]) * p[2])
        c1_h = c1_h_range[0] + round((c1_h_range[1] - c1_h_range[0]) * p[3])


        cropped_img0 = img[c0_h:c0_h+crop_size, c0_w:c0_w+crop_size, :]
        cropped_img1 = img[c1_h:c1_h+crop_size, c1_w:c1_w+crop_size, :]
        cropped_lab0 = lab[c0_h:c0_h+crop_size, c0_w:c0_w+crop_size]
        cropped_lab1 = lab[c1_h:c1_h+crop_size, c1_w:c1_w+crop_size]

        overlap_img0_h = max(c1_h-c0_h, 0)
        overlap_img1_h = max(c0_h-c1_h, 0)
        overlap_img0_w = max(c1_w-c0_w, 0)
        overlap_img1_w = max(c0_w-c1_w, 0)
        overlap_range_h = crop_size - np.abs(c0_h-c1_h)
        overlap_range_w = crop_size - np.abs(c0_w-c1_w)

        # h_start, h_end, w_start, w_end
        overlap_location0 = (overlap_img0_h, overlap_img0_h + overlap_range_h, \
                                overlap_img0_w, overlap_img0_w + overlap_range_w)
        overlap_location1 = (overlap_img1_h, overlap_img1_h + overlap_range_h, \
                                overlap_img1_w, overlap_img1_w + overlap_range_w)
        return cropped_img0, cropped_img1, cropped_lab0, cropped_lab1, overlap_location0, overlap_location1

    

    def random_resize(self, img, sp, width_factor: tuple, height_factor: tuple):
        p = np.random.rand(2)
        p_w, p_h = p[0], p[1]
        rescale_factor_w = (width_factor[1] - width_factor[0]) * p_w + width_factor[0]
        rescale_factor_h = (height_factor[1] - height_factor[0]) * p_h + height_factor[0]

        rescaled_w = int(img.shape[1] * rescale_factor_w)
        rescaled_h = int(img.shape[0] * rescale_factor_h)
        img_resized = cv2.resize(img, (rescaled_w, rescaled_h), interpolation = cv2.INTER_LINEAR)
        
        sp_resized = cv2.resize(sp, (rescaled_w, rescaled_h), interpolation = cv2.INTER_NEAREST)
        return img_resized, sp_resized

    def random_horizental_flip(self, img, sp, p_factor=0.5):
        p = np.random.rand(1)
        if p <= p_factor:
            img = cv2.flip(img, 1)
            sp = cv2.flip(sp, 1)
        return img, sp

    def get_mapper(self):
        fine_to_coarse27 = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24, 255: 255}
        fine_to_coarse_map = np.vectorize(lambda x: fine_to_coarse27[x])
        return fine_to_coarse_map

    def load_imdb(self):
        imdb = os.path.join(self.path_dataset, 'curated', '{}2017'.format(self.split), 'Coco164kFull_Stuff_Coarse_7.txt')
        imdb = tuple(open(imdb, "r"))
        imdb = [id_.rstrip() for id_ in imdb]
        return imdb

    def check_datapath_integrity(self, path_dataset):
        folders = os.listdir(path_dataset)
        assert 'curated'     in folders and \
               'images'      in folders and \
               'annotations' in folders, '{}'.format(folders)

        file_train_subject = os.path.join(path_dataset, r'curated/train2017/Coco164kFull_Stuff_Coarse_7.txt')
        file_val_subject   = os.path.join(path_dataset, r'curated/val2017/Coco164kFull_Stuff_Coarse_7.txt') 
        assert os.path.isfile(file_train_subject), file_train_subject
        assert os.path.isfile(file_val_subject), file_val_subject
        return file_train_subject, file_val_subject

    def __len__(self):
        return len(self.imdb)



if __name__ == '__main__':
    path_dataset = r'/storage/liumingyuan/dataset/coco/cocostuff/'
    dataset = CocoStuff27(path_dataset, split='val')
    sample = dataset.__getitem__(0)
    print(sample.keys())
    print(np.unique(sample['label']))



























