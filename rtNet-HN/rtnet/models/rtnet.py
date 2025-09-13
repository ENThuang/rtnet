import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import *
from .misc.backbone_utils import mobilenet_backbone, resnet_fpn_backbone
from .layers.feature_pyramid_network import LastLevelP6P7

from .layers.cca import CCA
from .layers.scr import SCR, SelfCorrelationComputation


class RtNet(nn.Module):
    def __init__(self, backbone="resnet18_3d", **kwargs):
        super().__init__()
        self.backbone = eval(backbone)(**kwargs)

    def forward(self, x):
        return self.backbone(x)


class RtNet25D(nn.Module):
    def __init__(self, backbone="resnet18_3d", slice_per_25d_group=3, **kwargs):
        super().__init__()
        self.slice_per_25d_group = slice_per_25d_group
        self.num_25d_group = kwargs["num_25d_group"]
        self.backbone = eval(backbone)(**kwargs)
        self.fuse25d = nn.Conv2d(
            self.backbone.classifier[0].in_features * kwargs["num_25d_group"],
            self.backbone.classifier[0].in_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):  # (bsz, mod, d, h, w)
        x = x.view(
            -1, self.slice_per_25d_group, x.shape[3], x.shape[4]
        )  # (bsz * group, 3, h, w)
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = x.view(
            -1, self.num_25d_group * x.shape[1], x.shape[2], x.shape[3]
        )  # (bsz, group * c, h, w)
        x = self.fuse25d(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x


class RtNet25DDualNet(nn.Module):
    def __init__(self, backbone="resnet18_3d", slice_per_25d_group=3, **kwargs):
        super().__init__()
        self.slice_per_25d_group = slice_per_25d_group
        self.num_25d_group = kwargs["num_25d_group"]
        pred_ene = kwargs["pred_ene"] if "pred_ene" in kwargs else False
        kwargs["num_classes"] += 1 if pred_ene else 0
        self.backbone_preserve = eval(backbone)(**kwargs)
        presv_feat_dim = self.backbone_preserve.classifier[0].in_features
        # self.backbone_preserve.classifier = nn.Identity()  # remove the classifier layer

        # self.backbone_invariant = eval(backbone)(**kwargs)
        from rtnet.models.backbones.mobilenetv3_2d import MobileNet_V3_Small_Weights

        kwargs["weights"] = MobileNet_V3_Small_Weights
        self.backbone_invariant = eval("mobilenet_v3_small")(**kwargs)
        invar_feat_dim = self.backbone_invariant.classifier[0].in_features
        # self.backbone_invariant.classifier = nn.Identity()  # remove the classifier layer

        self.fuse_pre = nn.Conv2d(
            presv_feat_dim * kwargs["num_25d_group"],
            presv_feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_invar = nn.Conv2d(
            invar_feat_dim * kwargs["num_25d_group"],
            invar_feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_all = nn.Conv2d(
            presv_feat_dim + invar_feat_dim,
            (presv_feat_dim + invar_feat_dim) // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.dropout = nn.Dropout(p=0.5)

        self.classifier = nn.Linear(
            (presv_feat_dim + invar_feat_dim) // 2, kwargs["num_classes"]
        )  # +1 for ENE prediction
        # self.classifier = nn.Linear(presv_feat_dim * 2, kwargs['num_classes'])

    def forward(self, x):  # (bsz, mod, d, h, w)
        x_preserve = x[:, :, 0]
        x_preserve = x_preserve.contiguous().view(
            -1, self.slice_per_25d_group, x_preserve.shape[3], x_preserve.shape[4]
        )  # (bsz * group, 3, h, w)

        x_invariant = x[:, :, 1]
        x_invariant = x_invariant.contiguous().view(
            -1, self.slice_per_25d_group, x_invariant.shape[3], x_invariant.shape[4]
        )  # (bsz * group, 3, h, w)

        # forward size-preserving and size-invariant
        x_preserve = self.backbone_preserve.features(x_preserve)
        x_preserve = self.backbone_preserve.avgpool(x_preserve)
        x_preserve = x_preserve.view(
            -1,
            self.num_25d_group * x_preserve.shape[1],
            x_preserve.shape[2],
            x_preserve.shape[3],
        )  # (bsz, group * c, h, w)
        x_preserve = self.fuse_pre(x_preserve)
        presv_out = torch.flatten(x_preserve, 1)
        # presv_out = self.dropout(presv_out)
        presv_out = self.backbone_preserve.classifier(presv_out)

        x_invariant = self.backbone_invariant.features(x_invariant)
        x_invariant = self.backbone_invariant.avgpool(x_invariant)
        x_invariant = x_invariant.view(
            -1,
            self.num_25d_group * x_invariant.shape[1],
            x_invariant.shape[2],
            x_invariant.shape[3],
        )  # (bsz, group * c, h, w)
        x_invariant = self.fuse_invar(x_invariant)
        invar_out = torch.flatten(x_invariant, 1)
        # invar_out = self.dropout(invar_out)
        invar_out = self.backbone_invariant.classifier(invar_out)

        # # fuse size-preserving and size-invariant features
        x_fused = self.fuse_all(torch.cat([x_preserve, x_invariant], dim=1))

        # # classifier
        x_fused = torch.flatten(x_fused, 1)
        # x_fused = self.dropout(x_fused)
        x_out = self.classifier(x_fused)

        # return x_out
        return torch.concat([presv_out, invar_out, x_out], dim=1)


class RtNet25DDualMILNet(nn.Module):
    def __init__(self, backbone="resnet18_3d", slice_per_25d_group=3, **kwargs):
        super().__init__()
        self.slice_per_25d_group = slice_per_25d_group
        self.num_25d_group = kwargs["num_25d_group"]

        self.backbone_preserve = eval(backbone)(**kwargs)
        presv_feat_dim = self.backbone_preserve.classifier[0].in_features
        presv_hid_dim = self.backbone_preserve.classifier[0].out_features
        self.backbone_preserve.meta_classifier = copy.deepcopy(
            self.backbone_preserve.classifier
        )
        self.backbone_preserve.ene_classifier = copy.deepcopy(
            self.backbone_preserve.classifier
        )
        self.backbone_preserve.maxpool = nn.AdaptiveMaxPool2d(1)
        self.backbone_preserve.classifier = nn.Identity()  # remove the classifier layer

        # self.backbone_invariant = eval(backbone)(**kwargs)
        from rtnet.models.backbones.mobilenetv3_2d import MobileNet_V3_Small_Weights

        kwargs["weights"] = MobileNet_V3_Small_Weights
        self.backbone_invariant = eval("mobilenet_v3_small")(**kwargs)
        invar_feat_dim = self.backbone_invariant.classifier[0].in_features
        invar_hid_dim = self.backbone_invariant.classifier[0].out_features
        self.backbone_invariant.meta_classifier = copy.deepcopy(
            self.backbone_invariant.classifier
        )
        self.backbone_invariant.ene_classifier = copy.deepcopy(
            self.backbone_invariant.classifier
        )
        self.backbone_invariant.maxpool = nn.AdaptiveMaxPool2d(1)
        self.backbone_invariant.classifier = (
            nn.Identity()
        )  # remove the classifier layer

        self.fuse_pre_meta = nn.Conv2d(
            presv_feat_dim * kwargs["num_25d_group"],
            presv_feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_pre_ene = nn.Conv2d(
            presv_feat_dim * kwargs["num_25d_group"],
            presv_feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_invar_meta = nn.Conv2d(
            invar_feat_dim * kwargs["num_25d_group"],
            invar_feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_invar_ene = nn.Conv2d(
            invar_feat_dim * kwargs["num_25d_group"],
            invar_feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_all_meta = nn.Conv2d(
            presv_feat_dim + invar_feat_dim,
            (presv_feat_dim + invar_feat_dim) // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_all_ene = nn.Conv2d(
            presv_feat_dim + invar_feat_dim,
            (presv_feat_dim + invar_feat_dim) // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.ene_classifier = nn.Linear(
            (presv_feat_dim + invar_feat_dim) // 2, kwargs["num_classes"]
        )
        self.meta_classifier = nn.Linear(
            (presv_feat_dim + invar_feat_dim) // 2, kwargs["num_classes"]
        )

        for name, m in self.named_children():
            if "backbone" in name:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # (bsz, mod, d, h, w)
        x_preserve = x[:, :, 0]
        x_preserve = x_preserve.contiguous().view(
            -1, self.slice_per_25d_group, x_preserve.shape[3], x_preserve.shape[4]
        )  # (bsz * group, 3, h, w)

        x_invariant = x[:, :, 1]
        x_invariant = x_invariant.contiguous().view(
            -1, self.slice_per_25d_group, x_invariant.shape[3], x_invariant.shape[4]
        )  # (bsz * group, 3, h, w)

        # forward size-preserving and size-invariant
        x_preserve = self.backbone_preserve.features(x_preserve)

        x_pres_meta = self.backbone_preserve.avgpool(x_preserve)
        x_pres_meta = x_pres_meta.view(
            -1,
            self.num_25d_group * x_pres_meta.shape[1],
            x_pres_meta.shape[2],
            x_pres_meta.shape[3],
        )  # (bsz, group * c, h, w)
        x_pres_meta = self.fuse_pre_meta(x_pres_meta)
        presv_meta_out = torch.flatten(x_pres_meta, 1)
        presv_meta_out = self.backbone_preserve.meta_classifier(presv_meta_out)

        x_pres_ene = self.backbone_preserve.maxpool(x_preserve)
        x_pres_ene = x_pres_ene.view(
            -1,
            self.num_25d_group * x_pres_ene.shape[1],
            x_pres_ene.shape[2],
            x_pres_ene.shape[3],
        )  # (bsz, group * c, h, w)
        x_pres_ene = self.fuse_pre_ene(x_pres_ene)
        presv_ene_out = torch.flatten(x_pres_ene, 1)
        presv_ene_out = self.backbone_preserve.ene_classifier(presv_ene_out)

        presv_out = torch.cat([presv_meta_out, presv_ene_out], dim=-1)

        x_invariant = self.backbone_invariant.features(x_invariant)
        x_invar_meta = self.backbone_invariant.avgpool(x_invariant)
        x_invar_meta = x_invar_meta.view(
            -1,
            self.num_25d_group * x_invar_meta.shape[1],
            x_invar_meta.shape[2],
            x_invar_meta.shape[3],
        )  # (bsz, group * c, h, w)
        x_invar_meta = self.fuse_invar_meta(x_invar_meta)
        invar_meta_out = torch.flatten(x_invar_meta, 1)
        invar_meta_out = self.backbone_invariant.meta_classifier(invar_meta_out)

        x_invar_ene = self.backbone_invariant.maxpool(x_invariant)
        x_invar_ene = x_invar_ene.view(
            -1,
            self.num_25d_group * x_invar_ene.shape[1],
            x_invar_ene.shape[2],
            x_invar_ene.shape[3],
        )  # (bsz, group * c, h, w)
        x_invar_ene = self.fuse_invar_ene(x_invar_ene)
        invar_ene_out = torch.flatten(x_invar_ene, 1)
        invar_ene_out = self.backbone_invariant.ene_classifier(invar_ene_out)

        invar_out = torch.cat([invar_meta_out, invar_ene_out], dim=-1)

        # # fuse size-preserving and size-invariant features
        x_fused_meta = self.fuse_all_meta(torch.cat([x_pres_meta, x_invar_meta], dim=1))
        x_fused_ene = self.fuse_all_ene(torch.cat([x_pres_ene, x_invar_ene], dim=1))

        # # classifier
        x_fused_meta = torch.flatten(x_fused_meta, 1)
        x_fused_ene = torch.flatten(x_fused_ene, 1)
        x_out_meta = self.meta_classifier(x_fused_meta)
        x_out_ene = self.ene_classifier(x_fused_ene)

        x_out = torch.cat([x_out_meta, x_out_ene], dim=-1)

        # return x_out
        return torch.concat([presv_out, invar_out, x_out], dim=1)


class RtNet25DDualMILNetV2(nn.Module):
    def __init__(self, backbone="resnet18_3d", slice_per_25d_group=3, **kwargs):
        super().__init__()
        self.slice_per_25d_group = slice_per_25d_group
        self.num_25d_group = kwargs["num_25d_group"]

        self.backbone_preserve = eval(backbone)(**kwargs)
        presv_feat_dim = self.backbone_preserve.classifier[0].in_features
        presv_hid_dim = self.backbone_preserve.classifier[0].out_features
        self.backbone_preserve.meta_classifier = copy.deepcopy(
            self.backbone_preserve.classifier
        )
        self.backbone_preserve.ene_classifier = copy.deepcopy(
            self.backbone_preserve.classifier
        )
        self.backbone_preserve.maxpool = nn.AdaptiveMaxPool2d(1)
        self.backbone_preserve.classifier = nn.Identity()  # remove the classifier layer

        # self.backbone_invariant = eval(backbone)(**kwargs)
        from rtnet.models.backbones.mobilenetv3_2d import MobileNet_V3_Small_Weights

        kwargs["weights"] = MobileNet_V3_Small_Weights
        self.backbone_invariant = eval("mobilenet_v3_small")(**kwargs)
        invar_feat_dim = self.backbone_invariant.classifier[0].in_features
        invar_hid_dim = self.backbone_invariant.classifier[0].out_features
        self.backbone_invariant.meta_classifier = copy.deepcopy(
            self.backbone_invariant.classifier
        )
        self.backbone_invariant.ene_classifier = copy.deepcopy(
            self.backbone_invariant.classifier
        )
        self.backbone_invariant.maxpool = nn.AdaptiveMaxPool2d(1)
        self.backbone_invariant.classifier = (
            nn.Identity()
        )  # remove the classifier layer

        self.fuse_pre_meta = nn.Conv2d(
            presv_feat_dim * kwargs["num_25d_group"],
            presv_feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_pre_ene = nn.Conv2d(
            presv_feat_dim * kwargs["num_25d_group"],
            presv_feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_invar_meta = nn.Conv2d(
            invar_feat_dim * kwargs["num_25d_group"],
            invar_feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_invar_ene = nn.Conv2d(
            invar_feat_dim * kwargs["num_25d_group"],
            invar_feat_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_all_meta = nn.Conv2d(
            presv_feat_dim + invar_feat_dim,
            (presv_feat_dim + invar_feat_dim) // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_all_ene = nn.Conv2d(
            presv_feat_dim + invar_feat_dim,
            (presv_feat_dim + invar_feat_dim) // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.ene_classifier = nn.Linear(
            (presv_feat_dim + invar_feat_dim) // 2, kwargs["num_classes"]
        )
        self.meta_classifier = nn.Linear(
            (presv_feat_dim + invar_feat_dim) // 2, kwargs["num_classes"]
        )

        for name, m in self.named_children():
            if "backbone" in name:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # (bsz, mod, d, h, w)
        x_preserve = x[:, :, 0]
        x_preserve = x_preserve.contiguous().view(
            -1, self.slice_per_25d_group, x_preserve.shape[3], x_preserve.shape[4]
        )  # (bsz * group, 3, h, w)

        x_invariant = x[:, :, 1]
        x_invariant = x_invariant.contiguous().view(
            -1, self.slice_per_25d_group, x_invariant.shape[3], x_invariant.shape[4]
        )  # (bsz * group, 3, h, w)

        # forward size-preserving and size-invariant
        x_preserve = self.backbone_preserve.features(x_preserve)

        x_pres_meta = self.backbone_preserve.avgpool(x_preserve)
        x_pres_meta = x_pres_meta.view(
            -1,
            self.num_25d_group * x_pres_meta.shape[1],
            x_pres_meta.shape[2],
            x_pres_meta.shape[3],
        )  # (bsz, group * c, h, w)
        x_pres_meta = self.fuse_pre_meta(x_pres_meta)
        presv_meta_out = torch.flatten(x_pres_meta, 1)
        presv_meta_out = self.backbone_preserve.meta_classifier(presv_meta_out)

        x_pres_ene = self.backbone_preserve.maxpool(x_preserve)
        x_pres_ene = x_pres_ene.view(
            -1,
            self.num_25d_group * x_pres_ene.shape[1],
            x_pres_ene.shape[2],
            x_pres_ene.shape[3],
        )  # (bsz, group * c, h, w)
        x_pres_ene = self.fuse_pre_ene(x_pres_ene)
        presv_ene_out = torch.flatten(x_pres_ene, 1)
        presv_ene_out = self.backbone_preserve.ene_classifier(presv_ene_out)

        presv_out = torch.cat([presv_meta_out, presv_ene_out], dim=-1)

        x_invariant = self.backbone_invariant.features(x_invariant)
        x_invar_meta = self.backbone_invariant.avgpool(x_invariant)
        x_invar_meta = x_invar_meta.view(
            -1,
            self.num_25d_group * x_invar_meta.shape[1],
            x_invar_meta.shape[2],
            x_invar_meta.shape[3],
        )  # (bsz, group * c, h, w)
        x_invar_meta = self.fuse_invar_meta(x_invar_meta)
        invar_meta_out = torch.flatten(x_invar_meta, 1)
        invar_meta_out = self.backbone_invariant.meta_classifier(invar_meta_out)

        x_invar_ene = self.backbone_invariant.maxpool(x_invariant)
        x_invar_ene = x_invar_ene.view(
            -1,
            self.num_25d_group * x_invar_ene.shape[1],
            x_invar_ene.shape[2],
            x_invar_ene.shape[3],
        )  # (bsz, group * c, h, w)
        x_invar_ene = self.fuse_invar_ene(x_invar_ene)
        invar_ene_out = torch.flatten(x_invar_ene, 1)
        invar_ene_out = self.backbone_invariant.ene_classifier(invar_ene_out)

        invar_out = torch.cat([invar_meta_out, invar_ene_out], dim=-1)

        # # fuse size-preserving and size-invariant features
        x_fused_meta = self.fuse_all_meta(torch.cat([x_pres_meta, x_invar_meta], dim=1))
        x_fused_ene = self.fuse_all_ene(torch.cat([x_pres_ene, x_invar_ene], dim=1))

        # # classifier
        x_fused_meta = torch.flatten(x_fused_meta, 1)
        x_fused_ene = torch.flatten(x_fused_ene, 1)
        x_out_meta = self.meta_classifier(x_fused_meta)
        x_out_ene = self.ene_classifier(x_fused_ene)

        x_out = torch.cat([x_out_meta, x_out_ene], dim=-1)

        # return x_out
        return torch.concat([presv_out, invar_out, x_out], dim=1)


class RtNet25DDualRENet(nn.Module):
    def __init__(
        self,
        backbone="resnet18_3d",
        slice_per_25d_group=3,
        temperature_attn=1.0,
        **kwargs
    ):
        super().__init__()
        self.slice_per_25d_group = slice_per_25d_group
        self.num_25d_group = kwargs["num_25d_group"]

        self.backbone_preserve = eval(backbone)(**kwargs)
        self.backbone_invariant = eval(backbone)(**kwargs)
        self.encoder_dim = self.backbone_preserve.classifier[0].in_features
        self.backbone_preserve.classifier = nn.Identity()  # remove the classifier layer
        self.backbone_invariant.classifier = (
            nn.Identity()
        )  # remove the classifier layer

        self.temperature_attn = temperature_attn
        self.scr_module = self._make_scr_layer(planes=[960, 96, 96, 96, 960])
        self.cca_module = CCA(kernel_sizes=[3, 3], planes=[16, 1])
        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(self.encoder_dim, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        self.meta_proj = nn.Sequential(
            nn.Linear(self.encoder_dim * 2, self.encoder_dim * 2, bias=False),
            # nn.BatchNorm1d(self.encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dim * 2, self.encoder_dim, bias=True),
        )

        self.ene_proj = nn.Sequential(
            nn.Linear(self.encoder_dim * 2, self.encoder_dim * 2, bias=False),
            # nn.BatchNorm1d(self.encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dim * 2, self.encoder_dim, bias=True),
        )

        self.meta_classifier = nn.Sequential(
            nn.Linear(
                in_features=self.encoder_dim,
                out_features=self.encoder_dim * 2,
                bias=True,
            ),
            nn.Hardswish(),
            # nn.Dropout(p=0.2, inplace=True),
            nn.Linear(
                in_features=self.encoder_dim * 2,
                out_features=kwargs["num_classes"],
                bias=True,
            ),
        )

        self.ene_classifier = nn.Sequential(
            nn.Linear(
                in_features=self.encoder_dim,
                out_features=self.encoder_dim * 2,
                bias=True,
            ),
            nn.Hardswish(),
            # nn.Dropout(p=0.2, inplace=True),
            nn.Linear(
                in_features=self.encoder_dim * 2,
                out_features=kwargs["num_classes"],
                bias=True,
            ),
        )

        for name, m in self.named_children():
            if "backbone" in name:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_scr_layer(self, planes):
        stride, kernel_size, padding = (
            (1, 1, 1),
            (5, 5),
            2,
        )  # TODO: check if kernal size is correct
        layers = list()

        corr_block = SelfCorrelationComputation(
            kernel_size=kernel_size, padding=padding
        )
        self_block = SCR(planes=planes, stride=stride)

        layers.append(corr_block)
        layers.append(self_block)
        return nn.Sequential(*layers)

    def get_4d_correlation_map(self, spt, qry):
        """
        The value H and W both for support and query is the same, but their subscripts are symbolic.
        :param spt: B * C * H_s * W_s
        :param qry: B * C * H_q * W_q
        :return: 4d correlation tensor: B * H_s * W_s * H_q * W_q
        :rtype:
        """
        batch_size = spt.shape[0]

        # reduce channel size via 1x1 conv
        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        similarity_map_einsum = torch.einsum("bcij,bckl->bijkl", spt, qry)
        return similarity_map_einsum

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def get_re_embedding(self, x_p, x_i):
        x_p = self.normalize_feature(x_p)  # b, c, h, w
        x_i = self.normalize_feature(x_i)

        # (B * C * Hs * Ws, B * C * Hq * Wq) -> B * Hs * Ws * Hq * Wq
        corr4d = self.get_4d_correlation_map(x_p, x_i)
        B, H_s, W_s, H_q, W_q = corr4d.size()

        # corr4d refinement
        corr4d = self.cca_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))
        corr4d_s = corr4d.view(B, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(B, H_s, W_s, H_q * W_q)

        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=1)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=3)

        # applying softmax for each side
        corr4d_s = F.softmax(corr4d_s / self.temperature_attn, dim=1)
        corr4d_s = corr4d_s.view(B, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.temperature_attn, dim=3)
        corr4d_q = corr4d_q.view(B, H_s, W_s, H_q, W_q)

        # suming up matching scores
        attn_s = corr4d_s.sum(dim=[3, 4])
        attn_q = corr4d_q.sum(dim=[1, 2])

        # applying attention
        x_p_attended = attn_s.unsqueeze(1) * x_p
        x_i_attended = attn_q.unsqueeze(1) * x_i

        x_p_attended_mean_pooled = x_p_attended.mean(dim=[-1, -2])
        x_i_attended_mean_pooled = x_i_attended.mean(dim=[-1, -2])
        x_p_attended_max_pooled = F.adaptive_max_pool2d(x_p_attended, (1, 1)).flatten(1)
        x_i_attended_max_pooled = F.adaptive_max_pool2d(x_i_attended, (1, 1)).flatten(1)

        x_p_out = torch.cat([x_p_attended_mean_pooled, x_p_attended_max_pooled], dim=-1)
        x_i_out = torch.cat([x_i_attended_mean_pooled, x_i_attended_max_pooled], dim=-1)

        return x_p_out, x_i_out

    def forward(self, x):  # (bsz, mod, d, h, w)
        x_preserve = x[:, :, 0]
        x_preserve = x_preserve.contiguous().view(
            -1, self.slice_per_25d_group, x_preserve.shape[3], x_preserve.shape[4]
        )  # (bsz * group, 3, h, w)

        x_invariant = x[:, :, 1]
        x_invariant = x_invariant.contiguous().view(
            -1, self.slice_per_25d_group, x_invariant.shape[3], x_invariant.shape[4]
        )  # (bsz * group, 3, h, w)

        # forward size-preserving and size-invariant
        x_p = self.backbone_preserve.features(x_preserve)
        identity_xp = x_p
        x_p = self.scr_module(x_p)  # b,c,h,w
        x_p = x_p + identity_xp
        x_p = F.relu(x_p, inplace=True)

        x_i = self.backbone_invariant.features(x_invariant)
        identity_xi = x_i
        x_i = self.scr_module(x_i)  # b,c,h,w
        x_i = x_i + identity_xi
        x_i = F.relu(x_i, inplace=True)

        embed_xp, embed_xi = self.get_re_embedding(x_p, x_i)  # g*b,2*c
        embed_xp_avg, embed_xp_max = (
            embed_xp[:, : self.encoder_dim],
            embed_xp[:, self.encoder_dim :],
        )
        embed_xi_avg, embed_xi_max = (
            embed_xi[:, : self.encoder_dim],
            embed_xi[:, self.encoder_dim :],
        )

        embed_meta = self.meta_proj(torch.cat([embed_xp_avg, embed_xi_avg], dim=1))
        embed_ene = self.ene_proj(torch.cat([embed_xp_max, embed_xi_max], dim=1))

        pooled_meta = torch.mean(
            embed_meta.view(-1, self.num_25d_group, embed_meta.shape[-1]), dim=1
        )
        pooled_ene = torch.mean(
            embed_ene.view(-1, self.num_25d_group, embed_ene.shape[-1]), dim=1
        )

        x_out_meta = self.meta_classifier(pooled_meta)
        x_out_ene = self.ene_classifier(pooled_ene)
        x_out = torch.cat([x_out_meta, x_out_ene], dim=-1)

        return x_out
        # return torch.concat([xp_out, xi_out, x_out], dim=1)


class RtNetWithFPN(nn.Module):
    def __init__(self, backbone="resnet18_3d", dropout=0.2, extra_head=None, **kwargs):
        super().__init__()
        if "mobilenet" in backbone:
            self.backbone = mobilenet_backbone(
                backbone_name=backbone,
                fpn=True,
                trainable_layers=6,
                returned_layers=[2, 3, 5],
                dropout=dropout,
                **kwargs
            )
        elif "resnet" in backbone:
            kwargs["in_channels"] = kwargs.pop("first_conv_channel")
            self.backbone = resnet_fpn_backbone(
                backbone_name=backbone, trainable_layers=5, **kwargs
            )
        else:
            raise NotImplementedError

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        num_return_layers = len(self.backbone.body.return_layers)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.out_channels * num_return_layers, 1664),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1664, kwargs["num_classes"]),
        )

    def forward(self, x):
        if not self.fpn:
            return self.backbone(x)
        else:
            x = self.backbone(x)
            x = torch.concat([p.flatten(1) for _, p in x.items()], dim=-1)
            x = self.classifier(x)

            return x

class RtNet_LDH_DualNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = MultiDualNet()
        
    def forward(self, x):  # (bsz, mod, d, h, w)
        x_preserve = x[:, :, 0]
        x_invariant = x[:, :, 1]
        output = self.backbone(x_preserve.squeeze(1), x_invariant.squeeze(1))

        return torch.concat([output, output, output], dim=1)