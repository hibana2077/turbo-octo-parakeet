import torch
import torch.nn as nn

from .backbones.vit_pytorch_uda import (
    uda_vit_base_patch16_224_TransReID,
    uda_vit_small_patch16_224_TransReID,
)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class build_uda_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super().__init__()
        _ = camera_num, view_num

        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.task_type = cfg.MODEL.TASK_TYPE
        self.in_planes = 384 if "small" in cfg.MODEL.Transformer_TYPE else 768

        if cfg.MODEL.TASK_TYPE == "classify_DA":
            self.base = factory[cfg.MODEL.Transformer_TYPE](
                img_size=cfg.INPUT.SIZE_CROP,
                aie_xishu=cfg.MODEL.AIE_COE,
                local_feature=cfg.MODEL.LOCAL_F,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                block_pattern=cfg.MODEL.BLOCK_PATTERN,
            )
        else:
            self.base = factory[cfg.MODEL.Transformer_TYPE](
                img_size=cfg.INPUT.SIZE_TRAIN,
                aie_xishu=cfg.MODEL.AIE_COE,
                local_feature=cfg.MODEL.LOCAL_F,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                use_cross=getattr(cfg.MODEL, "USE_CROSS", False),
                use_attn=getattr(cfg.MODEL, "USE_ATTN", True),
                block_pattern=cfg.MODEL.BLOCK_PATTERN,
            )

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if pretrain_choice == "imagenet":
            self.base.load_param(model_path)
            print("Loading pretrained ImageNet model......from {}".format(model_path))
        elif pretrain_choice == "un_pretrain":
            self.base.load_un_param(model_path)
            print("Loading trans_tune model......from {}".format(model_path))
        elif pretrain_choice == "pretrain":
            if model_path:
                self.load_param_finetune(model_path)
                print("Loading pretrained model......from {}".format(model_path))
            else:
                print("make model without initialization")

    def forward(
        self,
        x,
        x2,
        label=None,
        cam_label=None,
        view_label=None,
        domain_norm=False,
        return_logits=False,
        return_feat_prob=False,
        cls_embed_specific=False,
    ):
        _ = label
        inference_flag = not self.training
        base_outputs = self.base(
            x,
            x2,
            cam_label=cam_label,
            view_label=view_label,
            domain_norm=domain_norm,
            cls_embed_specific=cls_embed_specific,
            inference_target_only=inference_flag,
        )

        if isinstance(base_outputs, (tuple, list)) and len(base_outputs) == 4:
            global_feat, global_feat2, global_feat3, cross_attn = base_outputs
        elif isinstance(base_outputs, (tuple, list)) and len(base_outputs) == 2:
            global_feat, global_feat2 = base_outputs
            global_feat3, cross_attn = None, None
        else:
            raise RuntimeError("Unexpected backbone outputs from uda transformer")

        if self.neck == "":
            feat = global_feat
            feat2 = global_feat2
            feat3 = global_feat3
        else:
            feat = self.bottleneck(global_feat) if (self.training and global_feat is not None) else None
            feat2 = self.bottleneck(global_feat2) if global_feat2 is not None else None
            feat3 = self.bottleneck(global_feat3) if (self.training and global_feat3 is not None) else None

        if return_logits:
            cls_score = self.classifier(feat) if feat is not None else None
            cls_score2 = self.classifier(feat2) if feat2 is not None else None
            cls_score3 = self.classifier(feat3) if feat3 is not None else None
            return cls_score, cls_score2, cls_score3

        if self.training or return_feat_prob:
            cls_score = self.classifier(feat) if feat is not None else None
            cls_score2 = self.classifier(feat2) if feat2 is not None else None
            cls_score3 = self.classifier(feat3) if feat3 is not None else None
            return (
                (cls_score, global_feat, feat),
                (cls_score2, global_feat2, feat2),
                (cls_score3, global_feat3, feat3),
                cross_attn,
            )

        if self.neck_feat == "after" and self.neck != "":
            return feat, feat2, feat3
        return global_feat, global_feat2, global_feat3

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path, map_location="cpu")

        if isinstance(param_dict, dict) and "model" in param_dict:
            param_dict = param_dict["model"]
        elif isinstance(param_dict, dict) and "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]

        for k, v in param_dict.items():
            new_k = k.replace("module.", "") if "module." in k else k
            if new_k not in self.state_dict():
                continue
            self.state_dict()[new_k].copy_(v)


__factory_hh = {
    "uda_vit_small_patch16_224_TransReID": uda_vit_small_patch16_224_TransReID,
    "uda_vit_base_patch16_224_TransReID": uda_vit_base_patch16_224_TransReID,
}


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME != "transformer":
        raise NotImplementedError("Only transformer model is vendored in cdtrans_core")
    model = build_uda_transformer(num_class, camera_num, view_num, cfg, __factory_hh)
    print("===========building uda transformer===========")
    return model
