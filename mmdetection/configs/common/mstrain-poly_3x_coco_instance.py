_base_ = '../_base_/default_runtime.py'
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Use RepeatDataset to speed up training
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type='RepeatDataset',
#         times=3,
#         dataset=dict(
#             type=dataset_type,
#             ann_file=data_root + 'annotations/instances_train2017.json',
#             img_prefix=data_root + 'train2017/',
#             pipeline=train_pipeline)),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/instances_val2017.json',
#         img_prefix=data_root + 'val2017/',
#         pipeline=test_pipeline))
classes = ("Nóc xe ","Viền nóc (mui)","Pa vô lê","Kính chắn gió trước","Ca pô trước","Ca lăng","Lưới ca lăng","Ba đờ sốc trước","Lưới ba đờ sốc","Ốp ba đờ sốc trước","Lô gô","Đèn phản quang ba đờ sốc sau","Cốp sau / Cửa hậu","Ba đờ sốc sau","Ốp ba đờ sốc sau","Kính chắn gió sau","Đèn gầm","Ốp đèn  gầm","Cụm đèn trước","Mặt gương (kính) chiếu hậu","Vỏ gương (kính) chiếu hậu","Chân gương (kính) chiếu hậu","Đèn xi nhan trên gương chiếu hậu","Trụ kính trước","Tai (vè trước) xe","Ốp Tai (vè trước) xe","Đèn xi nhan ba đ sốc","Ốp đèn xi nhan ba đ sốc","Đèn hậu","Kính chết góc cửa","Kính hông (xe 7 chỗ)","Hông (vè sau) xe","Trụ kính sau","Đèn hậu trên cốp sau","La giăng (Mâm xe)","Lốp (vỏ) xe","Tay mở cửa","Kính cánh cửa","Cánh cửa","Móp, bẹp(thụng)","Nứt, rạn","Vỡ, thủng, rách","Trầy, xước","Trụ kính cánh cửa","Ốp hông (vè sau) xe","Nẹp cốp sau","Bậc cánh cửa","Nẹp ca pô trước",)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= '/home/a4000/huyenhc/Swin-Transformer-Object-Detection/dataset/data_2305/dataset_train.json',
        classes = classes,
        img_prefix= "/home/a4000/huyenhc/Swin-Transformer-Object-Detection/dataset/data_2305/data_remove_text",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/home/a4000/huyenhc/Swin-Transformer-Object-Detection/dataset/data_2305/dataset_test.json',
        img_prefix='/home/a4000/huyenhc/Swin-Transformer-Object-Detection/dataset/data_2305/data_remove_text',
        pipeline=test_pipeline,
        classes = classes,),
    test=dict(
        type=dataset_type,
        ann_file='/home/a4000/huyenhc/Swin-Transformer-Object-Detection/dataset/data_2305/dataset_test.json',
        img_prefix='/home/a4000/huyenhc/Swin-Transformer-Object-Detection/dataset/data_2305/data_remove_text',
        pipeline=test_pipeline,
        classes = classes,))
evaluation = dict(interval=1, metric=['bbox', 'segm'])

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
# Experiments show that using step=[9, 11] has higher performance
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
