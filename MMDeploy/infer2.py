import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np

import argparse
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel

from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import load_config
from mmdeploy.utils.device import parse_device_id
from mmdeploy.utils.timer import TimeCounter
from typing import List, Sequence, Tuple, Union
import cv2
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmdet.core import bbox2result
from mmdeploy.codebase.mmdet import get_post_processing_params, multiclass_nms
import sys
import numpy as np
import cv2
import nanoid
import os
from loguru import logger
import threading
import sys
sys.path.append("..")
from config.rules_base import ruleBase, postProcessMask
from config import process_config 
class TensorrtDetector:
    def __init__(self, deploy_cfg_path = "./configs/mmdet/instance-seg/instance-seg_tensorrt-int8_dynamic-320x320-1344x1344.py",
                       model_cfg_path = "../mmdetection/configs/insurance/cascade_mask_rcnn_restnext101.py",
                       model_file = ['end2end.engine']
):
        deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
        task_processor = build_task_processor(model_cfg, deploy_cfg, "cuda:0")
        self.model = task_processor.init_backend_model(model_file)
        self.model_cfg = model_cfg
        is_device_cpu = False
        device_id = None if is_device_cpu else parse_device_id("cuda:0")


        self.class_names = ["Nóc xe ","Viền nóc (mui)","Pa vô lê","Kính chắn gió trước","Ca pô trước","Ca lăng","Lưới ca lăng","Ba đờ sốc trước",
                            "Lưới ba đờ sốc","Ốp ba đờ sốc trước","Lô gô","Đèn phản quang ba đờ sốc sau","Cốp sau / Cửa hậu","Ba đờ sốc sau",
                            "Ốp ba đờ sốc sau","Kính chắn gió sau","Đèn gầm","Ốp đèn gầm","Cụm đèn trước","Mặt gương (kính) chiếu hậu",
                            "Vỏ gương (kính) chiếu hậu","Chân gương (kính) chiếu hậu","Đèn xi nhan trên gương chiếu hậu","Trụ kính trước",
                            "Tai (vè trước) xe","Ốp Tai (vè trước) xe","Đèn xi nhan ba đ sốc","Ốp đèn xi nhan ba đ sốc","Đèn hậu","Kính chết góc cửa",
                            "Kính hông (xe 7 chỗ)","Hông (vè sau) xe","Trụ kính sau","Đèn hậu trên cốp sau","La giăng (Mâm xe)","Lốp (vỏ) xe",
                            "Tay mở cửa","Kính cánh cửa","Cánh cửa","Móp, bẹp(thụng)","Nứt, rạn","Vỡ, thủng, rách","Trầy, xước","Trụ kính cánh cửa",
                            "Ốp hông (vè sau) xe","Nẹp cốp sau","Bậc cánh cửa","Nẹp ca pô trước"]
        
        self.damage_class_names = ("Móp, bẹp(thụng)", "Nứt, rạn", 'Vỡ, thủng, rách', "Trầy, xước")
        self.part_class_names = [x for x in self.class_names if x not in self.damage_class_names]
        self.cfg = process_config
        self.post_process_mask = postProcessMask()

    @staticmethod
    def get_label_idx(inputLabel):
        convertIdx =    {'Nóc xe ': 'hp_gY5jf8OpmvxvnhfUV7',
                        'Viền nóc (mui)': '9cZsnlDB-eg5WYS45BXL9',
                        'Pa vô lê': 'wqB6JKWVUQ2ng0Frmq9WP',
                        'Ốp pa vô lê': 'vpGUTzzgr4CI5q4Npf9Pl',
                        'Kính chắn gió trước': 'QFRWEnf4OnVevBzq-j3Vt',
                        'Ca pô trước': 'g2ma2BlHnrAg_dnAQmUMm',
                        'Nẹp Capo': 'Lb6q81yHqvKnR_a9X_eM5',
                        'Ca lăng': 'STc5BcShAHTbIa__7M1WW',
                        'Lưới ca lăng': 'PQP5-HHvPmzHO6woBe9af',
                        'Ba đờ sốc trước': 'AlzdY6KYppg0-_UWi2tKf',
                        'Lưới ba đờ sốc': 'VTK-CYfjDrEgrpukfS2kR',
                        'Ốp ba đờ sốc trước': 'bC22iFVpICTs4CdcLILtm',
                        'Nẹp ba đờ sốc trước': '3noU1M2Lv2LUwh8zBUrhQ',
                        'Lô gô': 'xNBaj39VUShzX9aV4rdis',
                        'Đèn phản quang ba đờ sốc sau': 'bEHFOPc7yzzIrnQ6mRlNY',
                        "Cốp sau / Cửa hậu": 'eT__vYUYckCtNa-pwbWmf',
                        'Cửa hậu': 'o6heaJ-X0sLlqwX-IMJHW',
                        'Ba đờ sốc sau': 'UKZabECsHGdSaz9qteAXV',
                        'Ốp ba đờ sốc sau': 'i50xW9ywl9J4d9wj3S7gU',
                        'Lưới ba đờ sốc sau': 'Usr8intynEV6hOASODFie',
                        'Nẹp ba đờ sốc sau': 'gM4Ul1ZL-0FdM9Kcqk4Pt',
                        'Kính chắn gió sau': 'VsuRTjwBJLco_jhD2dEqC',
                        'Nẹp cốp sau': 'PCQzJ7NLLo5IFeQrDkeYB',
                        'Đèn gầm': '3wMBixd4Cr-665X6oTVFN',
                        "Ốp đèn gầm": 'x3BYdT9VLsONs4GSE5_fK',
                        'Cụm đèn trước': 'mQdNrg4fALjVeDsKFWftr',
                        'Đèn xi nhan': 'NqnrpN5Hftyf-Xkq55cXY',
                        'Ốp đèn pha': 'bAmHt3yUzZEQRV3jf_6wo',
                        'Cụm gương (kính) chiếu hậu': 'bG7LxJsmX1prjmX_beKAe',
                        'Mặt gương (kính) chiếu hậu': 'AuAgDNopYZIxQc5792Ouu',
                        'Vỏ gương (kính) chiếu hậu': 'qpZ3X16gVxWSut6QbLeM_',
                        'Chân gương (kính) chiếu hậu': 'P1XnppziMPggPtHtFW1Ms',
                        'Đèn xi nhan trên gương chiếu hậu': '2kXZ5C1OTf6kTm19F6OLG',
                        'Kính chết góc cửa': '3d5g2LubM3UtU5URhlJw_',
                        'Trụ kính trước': '6rUWXivmQ1w97B7SHy8H-',
                        'Tai (vè trước) xe': 'p_MnYK9GqEWVDyF_0o2KC',
                        'Ốp Tai (vè trước) xe': 'Dg9sbp0hIFgulgNyY9wtb',
                        'Đèn xi nhan tai (vè ) xe': 'vSPFnIuYRmCZNapUMchRJ',
                        'Đèn xi nhan ba đ sốc': 'xeDJ1xEiCpGATe2U2ZllX',
                        'Ốp đèn xi nhan ba đ sốc': 'xeDJ1xEiCpGATe2U2ZllX',
                        'Ốp đèn xi nhan': 'wBBi0P8k6eUs1FdFl6SyD',
                        'Đèn hậu': 'GxxqDsW5Hd7_QpIv8HmcM',
                        'Ốp đèn hậu': 're1yAh3FZOk30RyHQRgB4',
                        'Kính chết góc cửa (xe 5 chỗ)': 'fIVHNQNIBZ4qqDhCBSQI_',
                        'Kính hông (xe 7 chỗ)': 'r6JzbF8ofkPFAMqSFh1G1',
                        'Hông (vè sau) xe': '3ISfP0W1bvZpiaQk7VGSe',
                        'Trụ kính sau': '8oFcsvuWTUWa-3Eh2fxsp',
                        'Đèn hậu trên cốp sau': '0Zw6DqUnUZx9RVVJAhAWE',
                        'La giăng (Mâm xe)': 'B4lqhSi80452IDyBjQvWW',
                        'Lốp (vỏ) xe': 'pUJdsMXhLOL2Cc2nwLUXw',
                        'Tay mở cửa': 'ftFsBinaVoJbPjL_VLfdH',
                        'Kính cánh cửa': 'gruqTIyLhQ7wrNCjPn5cO',
                        'Cánh cửa': '7ySLjGgEw9LuxsJqtURvF',
                        'Trụ kính cánh cửa': 'Ra9bXMb-ojxNm357IfpGt',
                        'Ốp hông (vè sau) xe': 'HZc1CUaxDPjbQOmHKNSV0',
                        'Bậc cánh cửa': '6KnkKysGbBW2SiCCj4Whg',
                        'Nẹp ca pô trước': 'HfRM5Yuy3HriODw1iCYh9',
                        'Móp, bẹp(thụng)': 'zmMJ5xgjmUpqmHd99UNq3',
                        'Nứt, rạn':'5IfgehKG297bQPLkYoZTw',
                        'Vỡ, thủng, rách':'wMxucuruHBUupNOoVy2MF',
                        'Trầy, xước':'yfMzer07THdYoCI1SM2LN'}

        if inputLabel in convertIdx.keys():
            return convertIdx[inputLabel]
        else:
            return inputLabel
    @staticmethod
    def __clear_outputs(
        test_outputs: List[Union[torch.Tensor, np.ndarray]]
    ) -> List[Union[List[torch.Tensor], List[np.ndarray]]]:
        batch_size = len(test_outputs[0])

        num_outputs = len(test_outputs)
        outputs = [[None for _ in range(batch_size)]
                   for _ in range(num_outputs)]

        for i in range(batch_size):
            inds = test_outputs[0][i, :, 4] > 0.0
            for output_id in range(num_outputs):
                outputs[output_id][i] = test_outputs[output_id][i, inds, ...]
        return outputs

    def preprocess(self, img):
        imgs = [img]
        is_batch = False
        cfg = self.model_cfg
        device = torch.device("cuda:0")
        if isinstance(imgs[0], np.ndarray):
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'


        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        test_pipeline = Compose(cfg.data.test.pipeline)
        datas = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                data = dict(img=img)
            else:
                data = dict(img_info=dict(filename=img), img_prefix=None)
            data = test_pipeline(data)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(imgs))
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        data = scatter(data, [device])[0]
        return data

    def post_processing1(self, data, outputs):
        input_img = data['img'][0].contiguous()
        img_metas = data['img_metas']
        batch_dets, batch_labels = outputs[:2]
        batch_masks = outputs[2] if len(outputs) == 3 else None
        batch_size = input_img.shape[0]
        img_metas = img_metas[0]
        results = []
        rescale = True
        for i in range(batch_size):
            dets, labels = batch_dets[i], batch_labels[i]
            if rescale:
                scale_factor = img_metas[i]['scale_factor']

                if isinstance(scale_factor, (list, tuple, np.ndarray)):
                    assert len(scale_factor) == 4
                    scale_factor = np.array(scale_factor)[None, :]  # [1,4]
                scale_factor = torch.from_numpy(scale_factor).to(dets)
                dets[:, :4] /= scale_factor

            if 'border' in img_metas[i]:
                x_off = img_metas[i]['border'][2]
                y_off = img_metas[i]['border'][0]
                dets[:, [0, 2]] -= x_off
                dets[:, [1, 3]] -= y_off
                dets[:, :4] *= (dets[:, :4] > 0)

            dets_results = bbox2result(dets, labels, len(self.model.CLASSES))

            if batch_masks is not None:
                masks = batch_masks[i]
                img_h, img_w = img_metas[i]['img_shape'][:2]
                ori_h, ori_w = img_metas[i]['ori_shape'][:2]
                export_postprocess_mask = True
                if self.model.deploy_cfg is not None:

                    mmdet_deploy_cfg = get_post_processing_params(
                        self.model.deploy_cfg)
                    # this flag enable postprocess when export.
                    export_postprocess_mask = mmdet_deploy_cfg.get(
                        'export_postprocess_mask', True)
                if not export_postprocess_mask:
                    masks = self.model.postprocessing_masks(
                        dets[:, :4], masks, ori_w, ori_h, self.model.device)
                else:
                    masks = masks[:, :img_h, :img_w]
                segms_results = [[] for _ in range(len(self.model.CLASSES))]
                for j in range(len(dets)):
                    segms_results[labels[j]].append(masks[j][0][0])
                results.append((dets_results, segms_results))
            else:
                results.append(dets_results)
            return results

    def predict(self, data):
        input_img = data['img'][0].contiguous()
        img_metas = data['img_metas']
        outputs = self.model.forward_test(input_img, img_metas, return_loss=False, rescale=True)
        outputs = TensorrtDetector.__clear_outputs(outputs)
        return outputs
    
    def __call__(self, img, type_mask = "gpu"):
        im_h, im_w, chanel = img.shape
        data = self.preprocess(img)
        outputs = self.predict(data)
        result = self.post_processing1(data, outputs)[0]
        # list_result = []
        classes = []
        scores = []
        boxes = []
        masks = []
        for c in range(len(result[0])):
            if len(result[0][c])!=0:
                for i in range(len(result[0][c])):
                    roi = np.array(result[0][c][i])[:4]
                    score = result[0][c][i][-1]
                    mask = result[1][c][i].int()[None, ...]
                    roi = np.array(roi).astype(np.int8)
                    classes.append(c)
                    scores.append(score)
                    boxes.append(roi)
                    masks.append(mask)
                    
        masks = torch.cat(masks, dim = 0)
        logger.info(f'FINISH SEGMENTATION! {masks.shape[0]} ANNOTATIONS WERE FOUND')
        scores= torch.from_numpy(np.array(scores)).cuda()
        boxes = torch.from_numpy(np.array(boxes)).cuda()
        
        segment_results = []
        def damage_masks_post_process():
            # save the damage masks
            index_score_damage = torch.where(scores > 0.3)[0]
            for class_name in self.damage_class_names:
                class_id = self.class_names.index(class_name)
                index_damage = np.where(np.array(classes) == class_id)[0]
                index_damage = [x for x in index_damage if x in index_score_damage]
                index_damage = torch.from_numpy(np.array(index_damage)).cuda()
                if len(index_damage) != 0:
                    mask_damage = torch.index_select(masks, 0, index_damage)
                    mask_damage = torch.sum(mask_damage, axis = 0)
                    mask_damage = (mask_damage > 0).int().cpu().numpy()
                    
                    # mask_img_path, mask_img_name = _save_mask( self.mask_tmp_dir, mask_damage,\
                    #     [0, 0, im_w, im_h], im_w, im_h)
                    segment_results.append({
                        'class_id': class_id,
                        'score': 1.0,
                        'box': [0.0, 0.0, 1.0, 1.0],
                        'box_abs': [0, 0, im_w, im_h],
                        'mask': mask_damage,
                        'is_part': 0
                    })
                else:
                    logger.info(f'Damage class {class_name} have no mask!')

            logger.info('FINISH MERGE CAR DAMAGE!')

        
        logger.info('START POSTPROCESS CAR DAMAGE!')
        y = threading.Thread(target=damage_masks_post_process)
        y.start()

        logger.info('START POSTPROCESS CAR PARTS!')
        index_item_part = []
        index_item_key_part = []
        index_key_part = []
        index_normal_part = []

        for i in range(len(scores)):
            class_name = self.class_names[classes[i]]
            if (scores[i] > 0.7) and (class_name in self.part_class_names):
                if (class_name in self.post_process_mask.listParts) and (class_name in self.post_process_mask.splitParts.keys()):
                    index_item_key_part.append(i)
                else:
                    if (class_name in self.post_process_mask.listParts):
                        index_item_part.append(i)
            
                    if (class_name in self.post_process_mask.splitParts.keys()):
                        index_key_part.append(i)
                    else:
                        index_normal_part.append(i) 
                                    
        index_item_key_part = torch.from_numpy(np.array(index_item_key_part)).cuda()
        index_item_part = torch.from_numpy(np.array(index_item_part)).cuda()
        index_key_part = torch.from_numpy(np.array(index_key_part)).cuda()
        index_normal_part = torch.from_numpy(np.array(index_normal_part)).cuda()

        masks_item_key_part_sum = torch.index_select(masks, 0, index_item_key_part).sum(axis = 0)
        masks_item_part_sum = torch.index_select(masks, 0, index_item_part).sum(axis = 0)

        masks_item_key_part = torch.index_select(masks, 0, index_item_key_part)
        masks_item_key_part = (masks_item_key_part - masks_item_part_sum > 0).int()

        masks_key_part = torch.index_select(masks, 0, index_key_part)
        masks_key_part = (masks_key_part - masks_item_key_part_sum - masks_item_part_sum> 0).int()

        masks_normal_part = torch.index_select(masks, 0, index_normal_part)

        masks_new_ori = torch.cat([masks_item_key_part, masks_key_part, masks_normal_part], dim = 0)
        new_index = torch.cat([index_item_key_part, index_key_part, index_normal_part]).int()
        scores_new = torch.index_select(scores, 0, new_index)
        classes_new = np.array(classes)[new_index.cpu().numpy()]
        bboxes_new =  torch.index_select(boxes, 0, new_index)
        masks_identical_all = (masks_new_ori*scores_new[..., None, None]).sum(axis = 0)

        masks_new = masks_new_ori * 2* scores_new[..., None, None] - masks_identical_all
        masks_new = (masks_new > 0).int()
        intersection_rate = masks_new.sum(axis = (1, 2))/masks_new_ori.sum(axis = (1, 2))

        def part_post_process(start, stop, step):
            for i in range(start, stop, step): # len(masks_new)
                mask = masks_new[i].cpu().numpy().astype(np.uint8)
                if intersection_rate[i] < 0.2:
                    continue
                score = scores_new[i]
                cls = classes_new[i]
                class_name = self.class_names[cls]
                box = bboxes_new[i].cpu().numpy().tolist()
                x1, y1, x2, y2 = box
                segment_results.append({
                                        'class_id': class_name,
                                        'score': float(score.cpu().numpy()),
                                        'box': [float(x1/im_w), float(y1/im_h), float(x2/im_w), float(y2/im_h)],
                                        'box_abs': [int(x1), int(y1), int(x2), int(y2)],
                                        'mask': mask,
                                        'is_part': 1
                                    })
    
        range_thead_1 =  int(len(masks_new) / 3)
        range_thead_2 =  int(len(masks_new) * 2 / 3)
        x1 = threading.Thread(target=part_post_process, args=(0, range_thead_1, 1))
        x1.start()

        x2 = threading.Thread(target=part_post_process, args=(range_thead_1, range_thead_2, 1))
        x2.start()

        x3 = threading.Thread(target=part_post_process, args=(range_thead_2, len(masks_new), 1))
        x3.start()

        x1.join()
        x2.join()
        x3.join()
        logger.info('FINISH POSTPROCESS CAR PARTS AND DAMAGE!')
        logger.info(segment_results)
        y.join()
        return segment_results
    def plot(self, img):
        data = self.preprocess(img)
        outputs = self.predict(data)
        results = self.post_processing1(data, outputs)
        self.model.show_result(img, results[0], score_thr=0.3)

def _save_mask(temp_dir, binary_mask, box, img_w, img_h):
    """
    Save the binary mask to file. The mask will be cropped by "box".
    binary_mask [512, 512] => The mask will be resized to (img_h, img_w) first.
    box: absolute box, format x1y1x2y2
    """
    temp_mask = np.ones(
        (binary_mask.shape[0], binary_mask.shape[1], 4), np.uint8)
    temp_mask *= 255
    binary_mask = binary_mask.astype(np.uint8)
    # temp_mask[:, :, 0] *= binary_mask
    # temp_mask[:, :, 1] *= binary_mask
    # temp_mask[:, :, 2] *= binary_mask
    temp_mask[:, :, 3] *= binary_mask

    # resize
    temp_mask = cv2.resize(temp_mask, (img_w, img_h))
    # crop
    temp_mask = temp_mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    temp_image_name = f"{nanoid.generate()}.png"
    mask_path = os.path.join(temp_dir, temp_image_name)

    cv2.imwrite(mask_path, temp_mask)

    return temp_image_name