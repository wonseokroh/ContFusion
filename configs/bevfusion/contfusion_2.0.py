_base_ = ['../_base_/default_runtime.py']

deterministic = False
find_unused_parameters = True

max_epoch=24
seed=0

pc_range =  [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.1, 0.1, 0.2]
image_size = [256, 704]

mean = [103.53, 116.28, 123.675]
std = [58.395, 57.12, 57.375]

augment2d = dict(
    resize = [[0.48, 0.48], [0.48, 0.48]],
    rotate = [0.0, 0.0],
    gridmask = dict(
        prob = 0.0,
        fixed_prob = True
    )
)

augment3d = dict(
    scale = [1.0, 1.0],
    rotate = [0.0, 0.0],
    translate = 0.0,
)

object_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
map_classes = ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider']

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

run_dir = 'results'
dataset_root = './data/nuscenes_mini/'
dataset_type = 'NuScenesDataset'

train_pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, reduce_beams=32, use_dim=5),
            dict(type='LoadPointsFromMultiSweeps', load_dim=5, pad_empty_sweeps=True, reduce_beams=32, remove_close=True, sweeps_num=9, use_dim=5),
            dict(type='LoadAnnotations3D', with_attr_label=False, with_bbox_3d=True, with_label_3d=True),
            dict(type='ObjectPaste',
                db_sampler=dict(
                    dataset_root=dataset_root, 
                    info_path=dataset_root + 'nuscenes_dbinfos_train.pkl',
                    points_loader=dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, reduce_beams=32, use_dim=5),
                    rate=1.0,
                    prepare=dict(
                        filter_by_difficulty=[-1], 
                        filter_by_min_points=dict(
                            car=5,
                            truck=5,
                            bus=5,
                            trailer=5,
                            construction_vehicle=5,
                            traffic_cone=5,
                            barrier=5,
                            motorcycle=5,
                            bicycle=5,
                            pedestrian=5)),
                    classes=object_classes, 
                    sample_groups=dict(
                        car=2,
                        truck=3,
                        construction_vehicle=7,
                        bus=4,
                        trailer=6,
                        barrier=2,
                        motorcycle=6,
                        bicycle=6,
                        pedestrian=2,
                        traffic_cone=2)),
                stop_epoch=-1),
            dict(type='ImageAug3D', bot_pct_lim=[0.0, 0.0], final_dim=[256, 704], is_train=True, rand_flip=False, resize_lim=[0.48, 0.48], rot_lim=[0.0, 0.0]),
            dict(type='GlobalRotScaleTrans', is_train=True, resize_lim=[1.0, 1.0], rot_lim=[0.0, 0.0], trans_lim=0.0),
            dict(type='LoadBEVSegmentation', classes=map_classes, dataset_root=dataset_root, xbound=[-50.0, 50.0, 0.5], ybound=[-50.0, 50.0, 0.5]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=pc_range),
            dict(type='ObjectRangeFilter', point_cloud_range=pc_range),
            dict(type='ObjectNameFilter', classes=object_classes),
            dict(type='ImageNormalize', mean=mean, std=std, to_rgb=True),
            dict(type='GridMask', fixed_prob=True, max_epoch=max_epoch, mode=1, offset=False, prob=0.0, ratio=0.5, rotate=1, use_h=True, use_w=True),
            dict(type='PointShuffle'),
            dict(type='DefaultFormatBundle3D', classes=object_classes),
            dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'], 
                meta_keys=['camera_intrinsics','camera2ego','lidar2ego','lidar2camera','lidar2image','img_aug_matrix','lidar_aug_matrix'])
]
test_pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, reduce_beams=32, use_dim=5),
            dict(type='LoadPointsFromMultiSweeps', load_dim=5, pad_empty_sweeps=True, reduce_beams=32, remove_close=True, sweeps_num=9, use_dim=5),
            dict(type='LoadAnnotations3D', with_attr_label=False, with_bbox_3d=True, with_label_3d=True),
            dict(type='ImageAug3D', bot_pct_lim=[0.0, 0.0], final_dim=[256, 704], is_train=False, rand_flip=False, resize_lim=[0.48, 0.48], rot_lim=[0.0, 0.0]),
            dict(type='GlobalRotScaleTrans', is_train=False, resize_lim=[1.0, 1.0], rot_lim=[0.0, 0.0], trans_lim=0.0),
            dict(type='LoadBEVSegmentation', classes=map_classes, dataset_root=dataset_root, xbound=[-50.0, 50.0, 0.5], ybound=[-50.0, 50.0, 0.5]),
            dict(type='PointsRangeFilter', point_cloud_range=pc_range),
            dict(type='ImageNormalize', mean=mean, std=std, to_rgb=True),
            dict(type='DefaultFormatBundle3D', classes=object_classes),
            dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev'], 
                meta_keys=['camera_intrinsics','camera2ego','lidar2ego','lidar2camera','lidar2image','img_aug_matrix','lidar_aug_matrix'])
]


data=dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file= dataset_root + 'nuscenes_infos_train.pkl',
        box_type_3d='LiDAR',
        dataset_root=dataset_root,
        map_classes=map_classes,
        modality=input_modality,
        object_classes=object_classes,
        pipeline=train_pipeline,
        test_mode=False,
        type=dataset_type,
        use_valid_flag=True),
    test=dict(
        ann_file= dataset_root + 'nuscenes_infos_val.pkl',
        box_type_3d='LiDAR',
        dataset_root=dataset_root,
        map_classes=map_classes,
        modality=input_modality,
        object_classes=object_classes,
        pipeline=test_pipeline,
        type=dataset_type),
    val=dict(
        ann_file= dataset_root + 'nuscenes_infos_val.pkl',
        box_type_3d='LiDAR',
        dataset_root=dataset_root,
        map_classes=map_classes,
        modality=input_modality,
        object_classes=object_classes,
        pipeline=test_pipeline,
        type=dataset_type)
)


model=dict(
    type='ContFusion',
    decoder=dict(
        backbone=dict(
            type='SECOND',
            conv_cfg=dict(type='Conv2d', bias=False),
            in_channels=256,
            layer_nums=[5, 5],     
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
            out_channels=[128, 256]),
        neck=dict(
            type='SECONDFPN', 
            in_channels=[128,256],
            norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
            out_channels=[256, 256],
            upsample_cfg=dict(type='deconv', bias= False),
            upsample_strides=[1, 2],
            use_conv_for_no_stride=True)),
    decoders=dict(
        img=dict(
            backbone=dict(
                type='SECOND',
                conv_cfg=dict(type='Conv2d', bias=False),
                in_channels=80,
                layer_nums=[5, 5],     
                layer_strides=[1, 2],
                norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
                out_channels=[128, 256]),
            neck=dict(
                type='SECONDFPN', 
                in_channels=[128,256],
                norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
                out_channels=[256, 256],
                upsample_cfg=dict(type='deconv', bias= False),
                upsample_strides=[1, 2],
                use_conv_for_no_stride=True)),
        pts=dict(
            backbone=dict(
                type='SECOND',
                conv_cfg=dict(type='Conv2d', bias=False),
                in_channels=256,
                layer_nums=[5, 5],     
                layer_strides=[1, 2],
                norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
                out_channels=[128, 256]),
            neck=dict(
                type='SECONDFPN', 
                in_channels=[128,256],
                norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
                out_channels=[256, 256],
                upsample_cfg=dict(type='deconv', bias= False),
                upsample_strides=[1, 2],
                use_conv_for_no_stride=True))),
    encoders=dict(
        camera=dict(
            backbone=dict(
                type='SwinTransformer',
                attn_drop_rate=0.0,
                convert_weights=True,
                depths=[2, 2, 6, 2],
                drop_path_rate=0.2,
                drop_rate=0.0,
                embed_dims=96,
                init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'),
                mlp_ratio=4,
                num_heads=[3, 6, 12, 24],
                out_indices=[1, 2, 3],
                patch_norm=True,
                qkv_bias=True,
                window_size=7,
                with_cp=False),
            neck=dict(
                type='GeneralizedLSSFPN',
                act_cfg=dict(type='ReLU', inplace=True),
                in_channels=[192, 384, 768],
                norm_cfg=dict(type='BN2d', requires_grad=True),
                num_outs=3,
                out_channels=256,
                start_level=0,
                upsample_cfg=dict(mode='bilinear', align_corners=False)),
            vtransform=dict(
                type='DepthLSSTransform',
                dbound=[1.0, 60.0, 0.5],
                downsample=2,
                feature_size=[32, 88],
                image_size=[256, 704],
                in_channels=256,
                out_channels=80,
                xbound=[-54.0, 54.0, 0.3],
                ybound=[-54.0, 54.0, 0.3],
                zbound=[-10.0, 10.0, 20.0])),
        lidar=dict(
            backbone=dict(
                type='SparseEncoder',
                block_type='basicblock',
                encoder_channels=[[16, 16, 32], [32, 32, 64], [64 ,64 ,128], [128, 128]],
                encoder_paddings=[[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]],
                in_channels=5,
                order=['conv', 'norm', 'act'],
                output_channels=128,
                sparse_shape=[1440, 1440, 41]),
            voxelize=dict(
                max_num_points=10,
                max_voxels=[120000, 160000],
                point_cloud_range=pc_range,
                voxel_size=[0.075, 0.075, 0.2]))),
    fuser=dict(
        type='ConvFuser',
        in_channels=[80, 256],
        out_channels=256),
    heads=dict(
        object=dict(
            type='ContFusionHead',
            activation= 'relu',
            auxiliary=True,
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                code_size=10,
                out_size_factor=8,
                pc_range=[-54, -54],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.0,
                voxel_size=[0.075, 0.075]),
            bn_momentum=0.1,
            common_heads=dict(
                center=[2, 2],
                dim=[3, 2],
                height=[1, 2],
                rot=[2, 2],
                vel=[2, 2]),
            dropout=0.1,
            ffn_channel=256,
            hidden_channel=128,
            in_channels=512,
            loss_bbox=dict(type='L1Loss', loss_weight=0.25, reduction='mean'),
            loss_cls=dict(type='FocalLoss', alpha=0.25, gamma=2.0, loss_weight=1.0, reduction='mean', use_sigmoid=True),
            loss_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0, reduction='mean'),
            nms_kernel_size=3,
            num_classes=10,
            num_decoder_layers=1,
            num_heads=8,
            num_proposals=200,
            test_cfg=dict(
                dataset='nuScenes',
                nms_type=None,
                grid_size=[1440, 1440, 41],
                out_size_factor=8,
                pc_range=[54.0, 54.0],
                voxel_size=[0.075, 0.075]),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner3D',
                    cls_cost=dict(type='FocalLossCost', alpha=0.25, gamma=2.0, weight=0.15),
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    iou_cost=dict(type='IoU3DCost', weight=0.25),
                    reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25)),
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                dataset='nuScenes',
                gaussian_overlap=0.1,
                grid_size=[1440, 1440, 41],
                min_radius=2,
                out_size_factor=8,
                point_cloud_range=pc_range,
                pos_weight=-1,
                voxel_size=[0.075, 0.075, 0.2],
                lidar_only=False
            )
        )
    )
)

evaluation=dict(interval=1, pipeline=test_pipeline)
fp16=dict(loss_scale=dict(growth_interval=2000))
gt_paste_stop_epoch=-1

# learning policy
lr_config=dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)
 
optimizer=dict(
    type='AdamW',
    lr=2.0e-4,
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'img_backbone': dict(lr_mult=0.1),
    #     }),
    weight_decay=0.01
)

optimizer_config=dict(
    grad_clip=dict(
        max_norm=35,
        norm_type=2)
)

runner=dict(
    max_epochs=max_epoch,
    type='CustomEpochBasedRunner'
)

log_config=dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
               type='WandbLoggerHook',
               init_kwargs=dict(
                   entity='letsgoeccv',
                   project='bevfusion',
                   name='mini-Fusion-Cont'
               )
        )
    ]
)


