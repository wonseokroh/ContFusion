from optparse import Values
from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization
from mmdet3d.models import FUSIONMODELS

from .base import Base3DFusionModel
from sklearn.manifold import TSNE
from mmdet3d.models.fusion_models.tsne_utils import visualize_tsne

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": Voxelization(**encoders["lidar"]["voxelize"]),
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.reshaper = torch.nn.Conv2d(80, 256, 1, stride=1, padding=0,).cuda()

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(256, 100),
            torch.nn.BatchNorm1d(64800),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 2),
            torch.nn.Sigmoid()
        )
        self.disc_loss = torch.nn.CrossEntropyLoss()

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)
        
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            f, c, n = self.encoders["lidar"]["voxelize"](res)
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        sizes = torch.cat(sizes, dim=0)

        if self.voxelize_reduce:
            feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
            feats = feats.contiguous()

        return feats, coords, sizes

    global tsne_features, tsne_labels
    tsne_features, tsne_labels = None, None

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in self.encoders:
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    img_metas,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            # Camera: torch.Size([1, 80, 180, 180])
            # LiDAR:  torch.Size([1, 256, 180, 180])
            features.append(feature)

        # cam_feat = self.reshaper(features[0])
        # cam_feat = cam_feat.squeeze().permute(1,2,0) # (180, 180, 256)
        # lidar_feat = features[1].squeeze().permute(1,2,0) # (180, 180, 256)

        # cam_feat = cam_feat.view([-1, 256]) # (32400, 256)
        # lidar_feat = lidar_feat.view([-1, 256]) # (32400, 256)        
        # disc_feats = torch.cat([cam_feat, lidar_feat], dim=0).unsqueeze(dim=0) # (1, 64800, 256)
        # disc_pred = self.discriminator(disc_feats) # (1, 64800, 2)
        
        # # Camera: 0 | LiDAR: 1
        # disc_gt = torch.cat([torch.zeros(32400, dtype=torch.long).cuda(), torch.ones(32400, dtype=torch.long).cuda()], dim=0) # (64800)
        # # print(f'pred: {disc_pred.shape} || gt: {disc_gt.shape}') (64800, 2) | (64800)
        # disc_loss = -1 * self.disc_loss(disc_pred[0], disc_gt)
        
        
        # # ============================= for t-sne =============================
        
        camera_tsne_feat, lidar_tsne_feat = features[0], features[1]
        
        camera_tsne_feat = self.reshaper(camera_tsne_feat)

        camera_tsne_feat = camera_tsne_feat.flatten()
        lidar_tsne_feat = lidar_tsne_feat.flatten()

        global tsne_features, tsne_labels
        if tsne_features is not None:
            tsne_features.append(camera_tsne_feat)
            tsne_features.append(lidar_tsne_feat)
            tsne_labels.append('camera')
            tsne_labels.append('lidar')
        else:
            tsne_features = [camera_tsne_feat, lidar_tsne_feat]
            tsne_labels = ['camera', 'lidar']
        tsne_features_np = torch.stack(tsne_features, 0).cpu().numpy()
        print(tsne_features_np.shape, '<========= tsne features np')
        if tsne_features_np.shape[0] >= 130:
            # print('============ draw tsne ============')
            tsne = TSNE(n_components=2).fit_transform(tsne_features_np)
            visualize_tsne(tsne, tsne_labels)

            ######################## Distibution ########################
            # import matplotlib.pyplot as plt
            # import numpy as np

            # plt.hist(tsne_features_np[1], density=True, alpha=0.3, range=[-0.5,0.5], color='blue', edgecolor='black') # LiDAR
            # plt.hist(tsne_features_np[0], density=True, alpha=0.3, range=[-0.5,0.5], color='red', edgecolor='black') # Camera

            # plt.savefig('./feature_disc_bar.png')
            # raise ValueError
            ######################## Distibution ########################

            raise ValueError
        
        # # ============================= for t-sne =============================
       
        

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            weight = 1
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, img_metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                        
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            # outputs['loss/object/loss_disc'] = disc_loss * weight

            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, img_metas)
                    bboxes = head.get_bboxes(pred_dict, img_metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
