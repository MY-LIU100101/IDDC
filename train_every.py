import sys
import os
import utils
import torch
from omegaconf import OmegaConf
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from tqdm import tqdm
import math

def train_one_epoch(cfg, global_epoch, model, train_loader, optimizer,scheduler, loss_func, logger, w0, w1, w2, w3):
    am = utils.AverageMeter(['l', 'l_pos', 'l_neg', 'l_bal', 'l_bal1', 'l_sup'])

    model['main'].train()
    model['linear'].train()

    for idx, batch in enumerate(train_loader):
        optimizer['main'].zero_grad()
        optimizer['linear'].zero_grad()

        img = batch['image'].cuda()
        img_aug = batch['image_aug'].cuda()
        lab = batch['label'].cuda()

        
        fea, code_inter, code = model['main'](img, backbone_trainable=False) 
        fea_aug, code_inter_aug, code_aug = model['main'](img_aug, backbone_trainable=False)
        #fea_color, code_inter_color, code_color = model['main'](img_color_aug)

        size_lab = lab.shape
        sup_head = model['linear'](torch.clone(code_inter.detach()))
        sup_head = F.interpolate(sup_head, size=size_lab[-2:], mode='bilinear', align_corners=True)
        loss_sup = loss_func['sup_loss'](sup_head, lab)
        
        loss_pos, loss_neg, loss_bal, loss_bal1 = loss_func['pos_loss'](\
                fea, fea_aug, code, code_aug, shift=w2, shift_neg=w3)
        #loss_neg = loss_func['neg_loss'](fea, code, shift=0.2)


        loss =  loss_pos+ w0*loss_neg + w1*loss_bal/50

        am.update('l', loss.item())
        am.update('l_neg', loss_neg.item())
        am.update('l_pos', loss_pos.item())
        am.update('l_bal', loss_bal.item())
        am.update('l_bal1', loss_bal1.item())
        am.update('l_sup', loss_sup.item())

        loss.backward()
        loss_sup.backward()
        
        optimizer['main'].step()
        optimizer['linear'].step()

        
        if idx % cfg.iter_display == 0:
            _t_lr = optimizer['main'].param_groups[1]['lr']
            logger.info('E:{}  I:{} lr:{:.2e} | L={:.6f}, L_pos={:.6f}, l_neg={:.6f}, l_bal={:.6f}, l_bal1={:.6f}, l_sup={:.6f}'\
                  .format(global_epoch, idx, _t_lr, am.pop('l'), am.pop('l_pos'), am.pop('l_neg'), am.pop('l_bal'), am.pop('l_bal1'), am.pop('l_sup')))


    scheduler['main'].step() #global_epoch
    scheduler['linear'].step()

def val(cfg, model, val_loader, logger, v_mode=2):
    model['main'].eval()
    model['linear'].eval()
    logger.info('Evaluating ... ...')

    hist_matrix_supervised = np.zeros((cfg.n_classes, cfg.n_classes))
    hist_matrix_cluster = np.zeros((cfg.n_classes, cfg.n_classes))

    all_unique_cat = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            if v_mode == 0:
                #if idx%10 == 1:
                #    print('{}'.format(idx))
                img = batch['patches'].squeeze(0).cuda()
                lab = batch['label']
                poses = batch['poses']

                size_lab = img.shape[-2:]

                
                
                _, output_inter, output = model['main'](img)
                output_sup = model['linear'](torch.clone(output_inter.detach()))

                

                pred_clu = F.interpolate(output, size=size_lab, mode='bilinear', align_corners=True).cpu()
                pred_clu = F.softmax(pred_clu, dim=1)
                pred_clu = utils.val_inv_window(pred_clu, batch['shape'], poses)
                pred_clu = pred_clu.argmax(dim=1).numpy()

                _unique_cat = np.unique(pred_clu)
                for _uu in _unique_cat:
                    if _uu not in all_unique_cat:
                        all_unique_cat.append(_uu)
                
                pred_sup = F.interpolate(output_sup, size=size_lab, mode='bilinear', align_corners=True).cpu()
                pred_sup = F.softmax(pred_sup, dim=1)
                pred_sup = utils.val_inv_window(pred_sup, batch['shape'], poses)
                pred_sup = pred_sup.argmax(dim=1).numpy()
            
            elif v_mode in [1, 2]:
                img = batch['image'].cuda()

                if cfg.val_mode == 2:
                    lab = batch['label']
                    size_lab = lab.shape[-2:]
                elif cfg.val_mode == 1:
                    lab = batch['label_ori']
                    size_lab = lab.shape[-2:]

                
                _, output_inter, output = model['main'](img)
                output_sup = model['linear'](torch.clone(output_inter.detach()))
                

                pred_clu = F.interpolate(output, size=size_lab, mode='bilinear', align_corners=True).cpu()
                pred_clu = F.softmax(pred_clu, dim=1)
                pred_clu = pred_clu.argmax(dim=1).numpy()

                _unique_cat = np.unique(pred_clu)
                for _uu in _unique_cat:
                    if _uu not in all_unique_cat:
                        all_unique_cat.append(_uu)
                
                pred_sup = F.interpolate(output_sup, size=size_lab, mode='bilinear', align_corners=True).cpu()
                pred_sup = F.softmax(pred_sup, dim=1)
                pred_sup = pred_sup.argmax(dim=1).numpy()
            else:
                raise ValueError()

            lab = np.array(lab)

            hist_matrix_cluster += utils.fast_hist(pred_clu.flatten(), lab.flatten(), cfg.n_classes)
            hist_matrix_supervised += utils.fast_hist(pred_sup.flatten(), lab.flatten(), cfg.n_classes)

    row_ind, col_ind = linear_sum_assignment(hist_matrix_cluster, maximize=True)
    new_hist_clu = np.zeros((cfg.n_classes, cfg.n_classes))
    for i in range(cfg.n_classes):
        new_hist_clu[col_ind[i]] = hist_matrix_cluster[i]
    results_clu = utils.get_result_metrics(new_hist_clu)

    resuts_sup = utils.get_result_metrics(hist_matrix_supervised)

    logger.info('####################################')
    logger.info('Clustering results: {}'.format(len(all_unique_cat)))
    logger.info('ACC  - All: {:.4f}'.format(results_clu['overall_precision (pixel accuracy)']))
    logger.info('mIOU - All: {:.4f}'.format(results_clu['mean_iou']))

    logger.info('Supervised results: ')
    logger.info('ACC  - All: {:.4f}'.format(resuts_sup['overall_precision (pixel accuracy)']))
    logger.info('mIOU - All: {:.4f}'.format(resuts_sup['mean_iou']))
    logger.info('####################################')
    return results_clu


def main(path_config):
    cfg = OmegaConf.load(path_config)
    OmegaConf.set_struct(cfg, False)

    w0, w1, w2, w3 = cfg.w0, cfg.w1, cfg.w2, cfg.w3
    ts_marker = '_w0_'+str(w0)+'_w1_'+str(w1)+'_w2_'+str(w2)+'_w3_'+str(w3)
    
    logger, cfg.output_path = utils.get_logger_and_folder(cfg.output_root, cfg.experiment_name, time_stamp=ts_marker)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.output_path, 'config_backup.yaml'))

    utils.set_seed(cfg.seed)
    utils.set_gpu(cfg.gpus)
    device = torch.device('cuda')


    train_dataset = utils.get_dataset(cfg.name_dataset, cfg.path_datadir, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                       batch_size=cfg.batch_size, 
                       shuffle=True, 
                       num_workers=cfg.num_workers,  
                       pin_memory=True, 
                       drop_last=True,)

    #if cfg.val_mode == 0:
    val_dataset_sw = utils.get_dataset(cfg.name_dataset, cfg.path_datadir, 'val', val_mode=0)
    val_dataloader_sw = torch.utils.data.DataLoader(val_dataset_sw, 
                       batch_size= 1, 
                       shuffle= False, 
                       num_workers= 0, 
                       pin_memory=False, 
                       drop_last=False,)#cfg.num_workers,  
    #elif cfg.val_mode in [1,2]:
    val_dataset = utils.get_dataset(cfg.name_dataset, cfg.path_datadir, 'val', val_mode=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                       batch_size= 32, 
                       shuffle= False, 
                       num_workers= cfg.num_workers, 
                       pin_memory=False, 
                       drop_last=False,)#cfg.num_workers,  

    model, linear_model = utils.get_model(cfg.model_exp, cfg=cfg) # , cluster_head

    optimizer_model = utils.get_optimizer_dif(model, lr=cfg.lr)
    
    optimizer_linear_model = utils.get_optimizer(linear_model, lr = cfg.lr_linear_model)


    max_iter = len(train_dataloader) * cfg.max_epochs
    print(max_iter)

    max_ep = cfg.max_epochs

    lambda1 = lambda ep: math.pow(1 - ep / max_ep, 0.9)
    scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=lambda1)
    scheduler_linear_model = torch.optim.lr_scheduler.LambdaLR(optimizer_linear_model, lr_lambda=lambda1)

    loss_contrastive = utils.get_cross_contrastive_loss()
    loss_neg = utils.get_neg_loss()
    loss_supervised_head = torch.nn.CrossEntropyLoss(ignore_index=255)

    best_iou_value = -1

    for epoch in range(cfg.max_epochs):
        
        train_one_epoch(cfg, 
                        global_epoch = epoch,
                        model = {'main': model,
                                 'linear': linear_model}, 
                        train_loader = train_dataloader,
                        optimizer = {'main':optimizer_model,
                                     'linear':optimizer_linear_model},
                        scheduler = {'main': scheduler_model,
                                     'linear': scheduler_linear_model},
                        loss_func = {'pos_loss': loss_contrastive, 
                                     'neg_loss': loss_neg,
                                     'sup_loss': loss_supervised_head},
                        logger = logger,
                        w0=w0, w1=w1, w2=w2, w3=w3)
        
        
        if epoch % 1 == 0:
            results = val(cfg, 
                model = {'main': model,
                         'linear': linear_model}, 
                val_loader = val_dataloader, 
                logger = logger,
                v_mode = 2)

            logger.info('E:{}, w0:{}, w1:{}, mIoU: {:.4f}'.format(epoch, w0, w1, results['mean_iou']))
        
            if results['mean_iou'] >= best_iou_value:
                best_iou_value = results['mean_iou']
                name_model_save = os.path.join(cfg.output_path, 'best_model.pth')
                #name_linear_save = os.path.join(cfg.output_path, 'best_linear.pth')
                torch.save({'classifier': model.classifier.state_dict(), 
                            'head_nonlinear': model.head_nonlinear.state_dict(),
                            'backbone': model.backbone.state_dict()}, name_model_save)
                #torch.save(linear_model.state_dict(), name_linear_save)
                logger.info('model best saved to {}......\n'.format(cfg.output_path))


    results = val(cfg, 
                model = {'main': model,
                         'linear': linear_model}, 
                val_loader = val_dataloader_sw, 
                logger = logger,
                v_mode = 0)
    name_model_save = os.path.join(cfg.output_path, 'last_model.pth')
    torch.save({'classifier': model.classifier.state_dict(), 
                'head_nonlinear': model.head_nonlinear.state_dict(),
                'backbone': model.backbone.state_dict()}, name_model_save)
    logger.info('model last saved to {}......\n'.format(cfg.output_path))
    

    torch.cuda.empty_cache()
        

if __name__ == '__main__':
    path_config = sys.argv[1]

    if not os.path.isfile(path_config):
        raise ValueError('Unexist config file {}'.format(path_config))
    main(path_config)
