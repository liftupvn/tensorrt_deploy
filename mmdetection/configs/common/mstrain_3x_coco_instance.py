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
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
classes = ("N??c xe ","Vi???n n??c (mui)","Pa v?? l??","K??nh ch???n gi?? tr?????c","Ca p?? tr?????c","Ca l??ng","L?????i ca l??ng","Ba ????? s???c tr?????c","L?????i ba ????? s???c","???p ba ????? s???c tr?????c","L?? g??","????n ph???n quang ba ????? s???c sau","C???p sau / C???a h???u","Ba ????? s???c sau","???p ba ????? s???c sau","K??nh ch???n gi?? sau","????n g???m","???p ????n  g???m","C???m ????n tr?????c","M???t g????ng (k??nh) chi???u h???u","V??? g????ng (k??nh) chi???u h???u","Ch??n g????ng (k??nh) chi???u h???u","????n xi nhan tr??n g????ng chi???u h???u","Tr??? k??nh tr?????c","Tai (v?? tr?????c) xe","???p Tai (v?? tr?????c) xe","????n xi nhan ba ?? s???c","???p ????n xi nhan ba ?? s???c","????n h???u","K??nh ch???t g??c c???a","K??nh h??ng (xe 7 ch???)","H??ng (v?? sau) xe","Tr??? k??nh sau","????n h???u tr??n c???p sau","La gi??ng (M??m xe)","L???p (v???) xe","Tay m??? c???a","K??nh c??nh c???a","C??nh c???a","M??p, b???p(th???ng)","N???t, r???n","V???, th???ng, r??ch","Tr???y, x?????c","Tr??? k??nh c??nh c???a","???p h??ng (v?? sau) xe","N???p c???p sau","B???c c??nh c???a","N???p ca p?? tr?????c",)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= './dataset/data_2305/dataset_train.json',
        classes = classes,
        img_prefix= "./dataset/data_2305/data_remove_text",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='./dataset/data_2305/dataset_test.json',
        img_prefix='./dataset/data_2305/data_remove_text',
        pipeline=test_pipeline,
        classes = classes,),
    test=dict(
        type=dataset_type,
        ann_file='./dataset/data_2305/dataset_test.json',
        img_prefix='./dataset/data_2305/data_remove_text',
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
