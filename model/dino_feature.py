import torch
import torch.nn as nn
import torch.nn.functional as F
import dino.vision_transformer as vits


class DinoWithHead(nn.Module):
    def __init__(self, cfg):
        super(DinoWithHead, self).__init__()
        self.cfg = cfg
        self.backbone = DinoFeaturizer(cfg)

        print(self.backbone.n_feats)

        self.head_nonlinear = nn.Sequential(
                        nn.Conv2d(self.backbone.n_feats, self.backbone.n_feats, (3, 3), padding=(1,1)),
                        nn.ReLU(True),
                        nn.Conv2d(self.backbone.n_feats, self.backbone.n_feats, (1, 1)),
                        nn.ReLU(True),
                        )
        self.classifier = nn.Conv2d(self.backbone.n_feats, cfg.n_classes, (1, 1))

    def forward(self, x, weight_factor = None, backbone_trainable=False):
        with torch.no_grad():
            features_drop, features_full = self.backbone(x)
            
        if weight_factor is not None:
            features_full = torch.einsum('bchw,c->bchw', features_full, weight_factor)

        codes_inter = self.head_nonlinear(features_full)
        codes = self.classifier(codes_inter)
        return features_full, codes_inter, codes


class DinoWithOutHead(nn.Module):
    def __init__(self, cfg):
        super(DinoWithOutHead, self).__init__()
        self.cfg = cfg
        self.backbone = DinoFeaturizer(cfg)
        print('using without head')


    def forward(self, x):
        with torch.no_grad():
            features_drop, features_full = self.backbone(x)
      
        return features_full, features_full, features_full


class DinoFeaturizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        #self.dim = dim
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768

        #self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        #self.relu = nn.Tanh()

    @staticmethod
    def _helper(x):
        # TODO remove this hard coded 56
        return F.interpolate(x, 56, mode="bilinear", align_corners=False)


    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)
        #code = self.relu(image_feat)
        code = image_feat
        return self.dropout(image_feat), code
