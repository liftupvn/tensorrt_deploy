from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import os
import timeit


config_file = '../work_dirs/self_sup_swin_tiny_roi32/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '../work_dirs/self_sup_swin_tiny_roi32/latest.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
imgs = [os.path.join('../data/coco/test/', f'{x}') for x in os.listdir('../data/coco/test/')]
# img = '../data/coco/test/image1641265283152304_2022012600048_THVO_TREN_noc_xe.jpg'

for img in tqdm(imgs):
    result = inference_detector(model, img)
    name = img.split('/')[-1].split('.')[0]
    model.show_result(img, result, out_file='roi32/' + f'{name}' + '.png') #  Save reasoning image 
# name = img.split('/')[-1].split('.')[0]
# result = inference_detector(model, img)
# model.show_result(img, result, out_file='tmp/' + f'{name}' + '.png')
