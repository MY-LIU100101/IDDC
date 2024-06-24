import sys
import os
import utils
import torch
from omegaconf import OmegaConf
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from tqdm import tqdm
# from crf import dense_crf

'''
def _apply_crf(tup, crf_weights):
    return dense_crf(tup[0], tup[1], crf_weights)


def batched_crf(pool, img_tensor, prob_tensor, crf_weights):
    #outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    b = img_tensor.shape[0]
    outputs = []
    img = img_tensor.detach().cpu()
    prob = prob_tensor.detach().cpu()
    for _b in range(b):
        outputs.append(_apply_crf( (img[_b, ...], prob[_b, ...]), crf_weights ))
    #outputs = _apply_crf( (img_tensor.detach().cpu(), prob_tensor.detach().cpu()) )
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)
'''

def val_ori(cfg, model, val_loader, val_mode=0, logger=None, crf_weights=None):
    model['main'].eval()
    print('Evaluating ... ...')

    hist_matrix_cluster = np.zeros((cfg.n_classes, cfg.n_classes))

    n_preds = np.zeros(shape = (27,))

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):

            if val_mode==0:
                ### sliding windows

                img = batch['patches'].squeeze(0).cuda()
                lab = batch['label']

                poses = batch['poses']
                ori_image = batch['image'].cuda()

                name = batch['img_name']
                name = os.path.basename(name[0])[:-3]

                size_lab = img.shape[-2:]

                _, output_inter, output0 = model['main'](img)
                _, _, output1 = model['main'](img.flip(dims=[3]))
                output = (output0 + output1.flip(dims=[3])) / 2


                pred_clu = F.interpolate(output, size=size_lab, mode='bilinear', align_corners=False)
                pred_clu = F.softmax(pred_clu, dim=1).cpu()
                #pred_clu = batched_crf(111, img, pred_clu, crf_weights).cpu() # pool
                pred_clu = utils.val_inv_window(pred_clu, batch['shape'], poses)
                pred_clu = pred_clu.argmax(dim=1, keepdim=True) #.numpy()

                n_preds += np.bincount(pred_clu.flatten(), weights=None, minlength=27)

                lab = F.interpolate(lab.unsqueeze(1).float(), size=(320,320), mode='nearest')
                pred_clu = F.interpolate(pred_clu.float(), size=(320,320), mode='nearest')

                lab.squeeze(1)
                pred_clu.squeeze(1)

                lab = lab.numpy().astype(int)
                pred_clu = pred_clu.numpy().astype(int)


            elif val_mode==1:
                ### Resize val at original shape
                img = batch['image'].cuda()
                lab = batch['label_ori']

                size_lab = lab.shape[-2:]

                _, output_inter, output = model['main'](img)
                output_sup = model['linear'](output_inter)

                pred_clu = F.interpolate(output, size=size_lab, mode='bilinear', align_corners=True).cpu()
                pred_clu = F.softmax(pred_clu, dim=1)
                pred_clu = pred_clu.argmax(dim=1).numpy()
                
                #pred_sup = F.interpolate(output_sup, size=size_lab, mode='bilinear', align_corners=True).cpu()
                #pred_sup = F.softmax(pred_sup, dim=1)
                #pred_sup = pred_sup.argmax(dim=1).numpy()

            elif val_mode==2:
                ### Resize val
                img = batch['image'].cuda()
                lab = batch['label']
                size_lab = lab.shape[-2:]

                _, output_inter, output = model['main'](img)
                #output_sup = model['linear'](output_inter)

                #print(output_inter.shape)
                #print(output.shape)
                # path_fea_save = r'/storage/liumingyuan/project/unseg/endend_ablative/vit_ibot_ab/output_interfea/fea/'
                # path_lab_save = r'/storage/liumingyuan/project/unseg/endend_ablative/vit_ibot_ab/output_interfea/lab/'
                #_fea = F.interpolate(output_inter, size=size_lab, mode='bilinear', align_corners=True).cpu().numpy().astype(np.float16)
                # _fea = output_inter.cpu().numpy()

                # _lab = F.interpolate(lab.unsqueeze(1).to(torch.float32), size=(20, 20), mode='nearest').numpy().astype(np.uint8)
                #print(_lab.shape)
                #print(_fea.shape)
                # np.save(os.path.join(path_fea_save, str(idx)+'.npy'), _fea)
                # np.save(os.path.join(path_lab_save, str(idx)+'.npy'), _lab)
                #assert 0 == 1

                aa = output.argmax(dim=1).cpu().numpy()
                n_preds += np.bincount(aa.flatten(), weights=None, minlength=27)

                pred_clu = F.interpolate(output, size=size_lab, mode='bilinear', align_corners=False).cpu()
                pred_clu = F.softmax(pred_clu, dim=1)
                pred_clu = pred_clu.argmax(dim=1).numpy()
                
                #pred_sup = F.interpolate(output_sup, size=size_lab, mode='bilinear', align_corners=False).cpu()
                #pred_sup = F.softmax(pred_sup, dim=1)
                #pred_sup = pred_sup.argmax(dim=1).numpy()

            else:
                raise ValueError()


            lab = np.array(lab)
            hist_matrix_cluster += utils.fast_hist(pred_clu.flatten(), lab.flatten(), cfg.n_classes)
            
    row_ind, col_ind = linear_sum_assignment(hist_matrix_cluster, maximize=True)
    new_hist_clu = np.zeros((cfg.n_classes, cfg.n_classes))
    for i in range(cfg.n_classes):
        new_hist_clu[col_ind[i]] = hist_matrix_cluster[i]
    results_clu = utils.get_result_metrics(new_hist_clu)

    logger.info('####################################')
    logger.info('Clustering results: ')
    logger.info('ACC  - All: {:.4f}'.format(results_clu['overall_precision (pixel accuracy)']))
    logger.info('mIOU - All: {:.4f}'.format(results_clu['mean_iou']))
    logger.info('####################################')
    return results_clu

def main(path_config, crf_weights=None):
    cfg = OmegaConf.load(path_config)
    OmegaConf.set_struct(cfg, False)

    logger, cfg.output_path = utils.get_logger_and_folder(cfg.path_log, cfg.experiment_name, for_val=True) #cfg.output_root

    utils.set_gpu(cfg.gpus)
    device = torch.device('cuda')

    if cfg.val_mode in [0]:
        val_dataset = utils.get_dataset(cfg.name_dataset, cfg.path_datadir, 'val', cfg.val_mode)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                           batch_size=1, 
                           shuffle=False, 
                           num_workers= 0, 
                           pin_memory=False, 
                           drop_last=False,)#cfg.num_workers,  
    elif cfg.val_mode in [1,2]:
        val_dataset = utils.get_dataset(cfg.name_dataset, cfg.path_datadir, 'val', cfg.val_mode)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                           batch_size= cfg.batch_size, 
                           shuffle=False, 
                           num_workers= 3, 
                           pin_memory=False, 
                           drop_last=False,)#cfg.num_workers,  
    else:
        raise ValueError()
    

    logger.info('Loading checkpoint from: {}'.format(cfg.load_checkpoint))
    model, _ = utils.get_model(cfg.model_exp, cfg=cfg)
    
    loaded_state_dict = torch.load(cfg.load_checkpoint)

    model.backbone.load_state_dict(loaded_state_dict['backbone'], strict=True)
    model.head_nonlinear.load_state_dict(loaded_state_dict['head_nonlinear'], strict=True)
    model.classifier.load_state_dict(loaded_state_dict['classifier'], strict=True)

    results = val_ori(cfg, 
        model = {'main': model,},
        val_loader = val_dataloader,
        val_mode = cfg.val_mode,
        logger = logger,
        crf_weights = crf_weights)

        
    torch.cuda.empty_cache()
        

if __name__ == '__main__':
    path_config = sys.argv[1]
    if not os.path.isfile(path_config):
        raise ValueError('Unexist config file {}'.format(path_config))

    
    #load_roots = \
    #    ['/storage1/liumingyuan/project/unsup_seg/rep_coco27/dino-s8-fromnode4/long_s_5_w0_1.401_w1_0.25_w2_0.2_w3_0.119_w4_0.0/',]

    #'/storage/liumingyuan/project/unsup_seg/eddc_e2e/output_coco1/node4/'
    
    '''
    MAX_ITER = int(sys.argv[2]) #2
    POS_W = int(sys.argv[3]) #9 #3
    POS_XY_STD = int(sys.argv[4]) #2 #1
    Bi_W = int(sys.argv[5]) #4 #4
    Bi_XY_STD = int(sys.argv[6]) #75 # 67
    Bi_RGB_STD = int(sys.argv[7]) #5 #3

    crf_weights = (MAX_ITER, POS_W, POS_XY_STD, Bi_W, Bi_XY_STD, Bi_RGB_STD)
    '''
    crf_weights = (0,)

    main(path_config, crf_weights=crf_weights)
