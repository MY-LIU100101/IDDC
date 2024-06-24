import logging
import os
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def get_logger_and_folder(path_output_root, experiment_name, time_stamp=None, for_val=False):
    if time_stamp is None:
        time_stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())[2:]

    if not for_val:
        path_output = os.path.join(path_output_root, experiment_name+'_'+time_stamp)
        if os.path.isdir(path_output):
            #raise ValueError('Logger folder {} already exist'.format(path_output))
            logger = get_logger(path_output)

        else:
            os.mkdir(path_output)
            logger = get_logger(path_output)
    else:
        path_output = path_output_root
        logger = get_logger(path_output, file_name= 'val_.log')


    return logger, path_output

def get_logger(path_logger, file_name='log.log'):
    #time_stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())[2:]
    #file_name = 'log.log'
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename= os.path.join(path_logger, file_name),
                    filemode='a')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-8s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger_main = logging.getLogger('main')
    return logger_main

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.enabled=False
    cudnn.deterministic = True
    cudnn.benchmark = False

def set_gpu(gpus: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def get_dataset(name_dataset, path_dataset, split, val_mode=0):
    assert val_mode in [0,1,2]
    if name_dataset == 'cocostuff27':
        from dataset.cocostuff import CocoStuff27
        dataset = CocoStuff27(path_dataset, 
                              split, val_mode=val_mode)
    elif name_dataset == 'cityscapes':
        from dataset.cityscapes import Cityscapes27
        dataset = Cityscapes27(path_dataset, split, val_mode=val_mode)
    elif name_dataset == 'cocostuff-full':
        from dataset.cocostuff_full import CocoStuffFull
        dataset = CocoStuffFull(path_dataset, split, val_mode=val_mode)
    else:
        raise ValueError('Unknown dataset {}'.format(name_dataset))
    return dataset

def get_model(name_model, cfg=None):
    if name_model == 'vit_base_dino_fixed_nonlinearhead':
        from model.dino_feature import DinoWithHead
        from model.linear_prob import LinearProb

        model = DinoWithHead(cfg)
        n_cls = cfg.n_classes
        model_linear = LinearProb(768, n_cls)

        model.train()
        model.cuda()

        model_linear.cuda()

        return model, model_linear

    elif name_model == 'vit_small_dino_fixed_nonlinearhead':
        from model.dino_feature import DinoWithHead
        from model.linear_prob import LinearProb

        model = DinoWithHead(cfg)

        n_cls = cfg.n_classes
        model_linear = LinearProb(384, n_cls)

        model.train()
        model.cuda()
        model_linear.cuda()
        return model, model_linear

    else:
        raise ValueError('Unkonwn dataset{}'.format(name_model))

def get_optimizer_dif(model, lr=5e-3):
    #optimizers = torch.optim.Adam(list(model.parameters()), lr=lr)
    optimizers = torch.optim.Adam([{'params': model.backbone.parameters(), 'lr': lr/1000.}, 
                                   {'params': model.classifier.parameters(), 'lr': lr}, 
                                   {'params': model.head_nonlinear.parameters(), 'lr': lr} ])
    return optimizers

def get_optimizer(model, lr=5e-3):
    optimizers = torch.optim.Adam(list(model.parameters()), lr=lr)
    return optimizers


def get_cross_contrastive_loss():
    from losses.contrast_loss import CrossContrastiveCorrelationLoss
    loss = CrossContrastiveCorrelationLoss()
    return loss

def get_neg_loss():
    from losses.contrast_loss import NegContrastiveLoss
    loss = NegContrastiveLoss()
    return loss

def val_inv_window(patches, img_size, poses):
    dim, c_h, c_w = patches.shape[1], patches.shape[2], patches.shape[3]
    img = torch.zeros(1, dim, img_size[0], img_size[1])
    count = torch.zeros(1, 1, img_size[0], img_size[1])

    for idx, p in enumerate(poses):
        img[0, :, p[0]:p[1], p[2]:p[3]] += patches[idx, ...]
        count[0, :, p[0]:p[1], p[2]:p[3]] += torch.ones(1, c_h, c_w)
    img = img / count

    return img


################################################################################

def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist
    
def get_result_metrics(histogram):
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp 

    iou = tp / (tp + fp + fn)
    prc = tp / (tp + fn) 
    opc = np.sum(tp) / np.sum(histogram)

    result = {"iou": iou,
              "mean_iou": np.nanmean(iou),
              "precision_per_class (per class accuracy)": prc,
              "mean_precision (class-avg accuracy)": np.nanmean(prc),
              "overall_precision (pixel accuracy)": opc}

    result = {k: 100*v for k, v in result.items()}
    return result

class AverageMeter():
    def __init__(self, keys: list = None):
        self.meters = {}
        for key in keys:
            self.meters[key] = []

    def update(self, key, value):
        assert key in list(self.meters.keys()), 'Unexist key {}'.format(key)
        self.meters[key].append(value)

    def pop(self, key):
        assert key in list(self.meters.keys()), 'Unexist key {}'.format(key)
        average_value = np.mean(self.meters[key])
        self.meters[key] = []
        return average_value