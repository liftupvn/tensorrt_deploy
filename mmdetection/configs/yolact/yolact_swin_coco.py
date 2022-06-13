_base_ = [
    './yolact_r50_1x8_coco.py',
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        upsample_cfg=dict(mode='bilinear')))

# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ("Nóc xe ","Pa vô lê","Kính chắn gió trước","Ca pô trước","Ca Lăng","Ba đờ sốc trước","Lô gô","Cốp sau","Ba đờ sốc sau","Kính chắn gió sau","Đèn gầm","Cụm đèn trước","Đèn cửa","Cụm kính chiếu hậu","Kính cánh cửa","Kính chết góc cửa","Cánh cửa","Cụm đèn hậu","Đèn lùi","Kính hông","Hông vè sau xe","Trụ kính sau","Trụ kính trước","Đèn xi nhan ba đờ sốc","Đèn phản quang","Vè trước xe","Nẹp ca lăng","Viền nóc xe","Móp (bẹp)","Nứt (rạn)","Vỡ, Thủng, Rách","Trầy (xước)",)
data = dict(
    train=dict(
        img_prefix='dataset/img',
        classes=classes,
        ann_file='dataset/train.json'),
    val=dict(
        img_prefix='dataset/img',
        classes=classes,
        ann_file='dataset/validation.json'),
    test=dict(
        img_prefix='dataset/img',
        classes=classes,
        ann_file='dataset/test.json'))