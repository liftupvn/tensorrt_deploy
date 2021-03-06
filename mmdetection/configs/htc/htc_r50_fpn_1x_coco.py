_base_ = './htc_without_semantic_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=183,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2))))
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# data = dict(
#     train=dict(
#         seg_prefix=data_root + 'stuffthingmaps/train2017/',
#         pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
dataset_type = 'CocoDataset'
classes = ("N??c xe ","Vi???n n??c (mui)","Pa v?? l??","K??nh ch???n gi?? tr?????c","Ca p?? tr?????c","Ca l??ng","L?????i ca l??ng","Ba ????? s???c tr?????c","L?????i ba ????? s???c","???p ba ????? s???c tr?????c","L?? g??","????n ph???n quang ba ????? s???c sau","C???p sau / C???a h???u","Ba ????? s???c sau","???p ba ????? s???c sau","K??nh ch???n gi?? sau","????n g???m","???p ????n  g???m","C???m ????n tr?????c","M???t g????ng (k??nh) chi???u h???u","V??? g????ng (k??nh) chi???u h???u","Ch??n g????ng (k??nh) chi???u h???u","????n xi nhan tr??n g????ng chi???u h???u","Tr??? k??nh tr?????c","Tai (v?? tr?????c) xe","???p Tai (v?? tr?????c) xe","????n xi nhan ba ?? s???c","???p ????n xi nhan ba ?? s???c","????n h???u","K??nh ch???t g??c c???a","K??nh h??ng (xe 7 ch???)","H??ng (v?? sau) xe","Tr??? k??nh sau","????n h???u tr??n c???p sau","La gi??ng (M??m xe)","L???p (v???) xe","Tay m??? c???a","K??nh c??nh c???a","C??nh c???a","M??p, b???p(th???ng)","N???t, r???n","V???, th???ng, r??ch","Tr???y, x?????c","Tr??? k??nh c??nh c???a","???p h??ng (v?? sau) xe","N???p c???p sau","B???c c??nh c???a","N???p ca p?? tr?????c",)
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
