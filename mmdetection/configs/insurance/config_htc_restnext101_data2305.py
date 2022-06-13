
# The new config inherits a base config to highlight the necessary modification
# _base_ = './cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
# _base_ = "/home/a4000/Swin-Transformer-Object-Detection/faster_rcnn_r50_fpn_1x_coco.py"
# _base_ = '/home/a4000/huyenhc/Swin-Transformer-Object-Detection/pretrained/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
_base_ = '../htc/htc_without_semantic_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
runner = dict(type='EpochBasedRunner', max_epochs=20)

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=32),
#         mask_head=dict(num_classes=32)))

# Modify dataset related settings
# dataset_type = 'COCODataset'
# classes = ("Nóc xe ","Pa vô lê","Kính chắn gió trước","Ca pô trước","Ca Lăng","Ba đờ sốc trước","Lô gô","Cốp sau","Ba đờ sốc sau","Kính chắn gió sau","Đèn gầm","Cụm đèn trước","Đèn cửa","Cụm kính chiếu hậu","Kính cánh cửa","Kính chết góc cửa","Cánh cửa","Cụm đèn hậu","Đèn lùi","Kính hông","Hông vè sau xe","Trụ kính sau","Trụ kính trước","Đèn xi nhan ba đờ sốc","Đèn phản quang","Vè trước xe","Nẹp ca lăng","Viền nóc xe","Móp (bẹp)","Nứt (rạn)","Vỡ, Thủng, Rách","Trầy (xước)",)
# data = dict(
#     train=dict(
#         img_prefix='dataset/img',
#         classes=classes,
#         ann_file='dataset/train.json'),
#     val=dict(
#         img_prefix='dataset/img',
#         classes=classes,
#         ann_file='dataset/validation.json'),
#     test=dict(
#         img_prefix='dataset/img',
#         classes=classes,
#         ann_file='dataset/test.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'pretrained/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth'
# load_from = '/home/a4000/Swin-Transformer-Object-Detectio/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'