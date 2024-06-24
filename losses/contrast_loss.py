import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossContrastiveCorrelationLoss(nn.Module):
    def __init__(self):
        super(CrossContrastiveCorrelationLoss, self).__init__()

    def forward(self, raw_fea0, raw_fea1, fine_code0, fine_code1, shift, shift_neg, flag_assert = 0):

        ## sampling
        batch_size = raw_fea0.shape[0]
        fine_code0 = F.softmax(fine_code0, dim=1)
        fine_code1 = F.softmax(fine_code1, dim=1)

        raw_fea0 = F.normalize(raw_fea0, dim=1)
        raw_fea1 = F.normalize(raw_fea1, dim=1)
        #raw_fea0 = self.normalize_batch(raw_fea0)
        #raw_fea1 = self.normalize_batch(raw_fea1)
        #assert 0 == 1

        neg_rand_idx = np.random.randint(low=1, high=batch_size-1, size=3)

        raw_fea_neg0 = torch.cat((raw_fea0[neg_rand_idx[0]:, ...], raw_fea0[0:neg_rand_idx[0], ...]), dim=0)
        fine_code_neg0 = torch.cat((fine_code0[neg_rand_idx[0]:, ...], fine_code0[0:neg_rand_idx[0], ...]), dim=0)
        
        raw_fea_neg1 = torch.cat((raw_fea0[neg_rand_idx[1]:, ...], raw_fea0[0:neg_rand_idx[1], ...]), dim=0)
        fine_code_neg1 = torch.cat((fine_code0[neg_rand_idx[1]:, ...], fine_code0[0:neg_rand_idx[1], ...]), dim=0)

        raw_fea_neg2 = torch.cat((raw_fea0[neg_rand_idx[2]:, ...], raw_fea0[0:neg_rand_idx[2], ...]), dim=0)
        fine_code_neg2 = torch.cat((fine_code0[neg_rand_idx[2]:, ...], fine_code0[0:neg_rand_idx[2], ...]), dim=0)

        with torch.no_grad():
            f_cor_pos0 = self.tensor_correlation(raw_fea0, raw_fea1)
            f_cor_neg0 = self.tensor_correlation(raw_fea0, raw_fea_neg0)
            f_cor_neg1 = self.tensor_correlation(raw_fea0, raw_fea_neg1)
            f_cor_neg2 = self.tensor_correlation(raw_fea0, raw_fea_neg2)
            

        c_cor_pos0 = self.tensor_correlation(fine_code0, fine_code1)
        c_cor_pos0 = torch.mul(torch.exp((1-c_cor_pos0)), f_cor_pos0.clamp(min=1e-8))

        #c_cor_pos_self = self.tensor_correlation(fine_code0, fine_code0)
        #c_cor_pos_self = torch.mul(torch.exp((1-c_cor_pos_self)), f_cor_pos_self.clamp(min=0))

        #c_cor_pos0 = (1-c_cor_pos0)

        pos_losses = c_cor_pos0[f_cor_pos0>shift].mean()

        #pos_losses_self = c_cor_pos_self[f_cor_pos_self>shift].mean()
        #pos_losses = (pos_losses0+pos_losses_self)/2.
        

        c_cor_neg0 = self.tensor_correlation(fine_code0, fine_code_neg0)
        c_cor_neg1 = self.tensor_correlation(fine_code0, fine_code_neg1)
        c_cor_neg2 = self.tensor_correlation(fine_code0, fine_code_neg2)
        

        c_cor_neg0 = torch.mul(torch.exp(c_cor_neg0), 1-f_cor_neg0.clamp(min=0))
        c_cor_neg1 = torch.mul(torch.exp(c_cor_neg1), 1-f_cor_neg1.clamp(min=0))
        c_cor_neg2 = torch.mul(torch.exp(c_cor_neg2), 1-f_cor_neg2.clamp(min=0))


        neg_loss0 = c_cor_neg0[ f_cor_neg0<shift_neg ].mean()
        neg_loss1 = c_cor_neg1[ f_cor_neg1<shift_neg ].mean()
        neg_loss2 = c_cor_neg2[ f_cor_neg2<shift_neg ].mean()
     

        neg_losses = (neg_loss0+neg_loss1+neg_loss2)/3
        

        x = fine_code0.mean((0,2,3))
        x = x / torch.max(x) + 1e-6

        wb_l, wb_k = 5, 0.3
        balance_losses_wb = wb_k/wb_l* torch.pow(x/wb_l, wb_k-1.) * \
                         torch.exp(-1*torch.pow(x/wb_l, wb_k))
        wb1 = wb_k/wb_l* (1./wb_l)**(wb_k-1.) * np.exp(-1* (1./wb_l)**(wb_k))
        balance_losses_wb = (balance_losses_wb).mean()

        fine_code0 = fine_code0.permute((1,2,3,0)).mean((1,2,3))
        balance_losses0 = fine_code0 * torch.log(fine_code0.clamp(1e-6))
        balance_losses = balance_losses0.mean()

        return pos_losses, neg_losses, balance_losses_wb, balance_losses

    def normalize_batch(self, x):
        with torch.no_grad():
            x1 = torch.pow(x, 2)
            x1 = x1.sum((2, 3))
            x1 = torch.sqrt(x1)
            x1, _ = torch.max(x1, dim=1)
            b = x1.shape[0]
            x1 = x1.view((b,1,1,1))
            #print(x1.shape)
            #print(x.shape)
        return torch.div(x, x1)

    def tensor_correlation(self, a, b):
        return torch.einsum("nchw,ncij->nhwij", a, b)

    def sample(self, t: torch.Tensor, coords: torch.Tensor):
        return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)

class NegContrastiveLoss(nn.Module):
    def __init__(self):
        super(NegContrastiveLoss, self).__init__()

    def forward(self, raw_fea0, fine_code0, shift):

        ## sampling
        batch_size = raw_fea0.shape[0]
        fea_size = raw_fea0.shape[1]

        raw_fea0 = F.normalize(raw_fea0, dim=1)
        fine_code0 = F.softmax(fine_code0, dim=1)

        raw_fea1 = torch.cat((raw_fea0[1:, ...], raw_fea0[0, ...].unsqueeze(0)), dim=0)
        fine_code1 = torch.cat((fine_code0[1:, ...], fine_code0[0, ...].unsqueeze(0)), dim=0)

        raw_fea2 = torch.cat((raw_fea0[2:, ...], raw_fea0[0:2, ...]), dim=0)
        fine_code2 = torch.cat((fine_code0[2:, ...], fine_code0[0:2, ...]), dim=0)

        

        #raw_fea2 = torch.cat((raw_fea0[2:, ...], raw_fea0[0:2, ...]), dim=0)
        #fine_code2 = torch.cat((fine_code0[2:, ...], fine_code0[0:2, ...]), dim=0)

        with torch.no_grad():
            f_cor1 = self.tensor_correlation(raw_fea0, raw_fea1)
            f_cor2 = self.tensor_correlation(raw_fea0, raw_fea2)
            
        c_cor1 = self.tensor_correlation(fine_code0, fine_code1)
        c_cor2 = self.tensor_correlation(fine_code0, fine_code2)
        #c_cor3 = self.tensor_correlation(fine_code0, fine_code3)

        c_cor1 = torch.exp(c_cor1) * torch.exp(1-f_cor1)
        c_cor2 = torch.exp(c_cor2) * torch.exp(1-f_cor2)
        #c_cor3 = torch.exp(c_cor3) * torch.exp(1-f_cor3)

        _loss = c_cor1[f_cor1<shift].mean() + c_cor2[f_cor2<shift].mean() #+ c_cor3[f_cor3<shift].mean()

        all_losses = _loss / 2

        return all_losses

    def tensor_correlation(self, a, b):
        return torch.einsum("nchw,ncij->nhwij", a, b)