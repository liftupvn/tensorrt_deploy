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
import tritonclient.grpc as grpcclient
#import tritonclient.http as httpclient

from tritonclient.utils import InferenceServerException, triton_to_np_dtype
import tritonclient.utils.shared_memory as shm
import sys

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

        self.triton_client = grpcclient.InferenceServerClient(
            url="192.168.81.111:8001",
            verbose=False,
            ssl=False
            #root_certificates=FLAGS.root_certificates,
            #private_key=FLAGS.private_key,
            #certificate_chain=FLAGS.certificate_chain
	)
        self.model_name = 'yolox'
        # Health check
        if not self.triton_client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)

        if not self.triton_client.is_model_ready(self.model_name):
            print("FAILED : is_model_ready")
            sys.exit(1)

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

                        #Damage uuid
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
        """Removes additional outputs and detections with zero and negative
        score.

        Args:
            test_outputs (List[Union[torch.Tensor, np.ndarray]]):
                outputs of forward_test.

        Returns:
            List[Union[List[torch.Tensor], List[np.ndarray]]]:
                outputs with without zero score object.
        """
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
        print(cfg.data.test.pipeline)
        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
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
                    scale_factor = np.array(scale_factor)[None, :]
  # [1,4]  
                # import pdb; pdb.set_trace()
                # scale_factor = torch.from_numpy(scale_factor).to(dets)
                scale_factor = torch.from_numpy(scale_factor)
                dets[:, :4] /= scale_factor

            if 'border' in img_metas[i]:
                # offset pixel of the top-left corners between original image
                # and padded/enlarged image, 'border' is used when exporting
                # CornerNet and CentripetalNet to onnx
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
                # avoid to resize masks with zero dim
                # if rescale and masks.shape[0] != 0:
                #     masks = torch.nn.functional.interpolate(
                #         masks.unsqueeze(0), size=(ori_h, ori_w))
                #     masks = masks.squeeze(0)
                # if masks.dtype != bool:
                #     masks = masks >= 0.5
                # aligned with mmdet to easily convert to numpy
                # masks = masks.cpu()
                segms_results = [[] for _ in range(len(self.model.CLASSES))]
                for j in range(len(dets)):
                    segms_results[labels[j]].append(masks[j][0][0])
                results.append((dets_results, segms_results))
            else:
                results.append(dets_results)
            return results

    def predict(self, data):
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('input', [1, 3, 588, 800], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('masks'))
        outputs.append(grpcclient.InferRequestedOutput('dets'))
        outputs.append(grpcclient.InferRequestedOutput('labels'))
        input_img = data['img'][0].contiguous()
        img_metas = data['img_metas']
        input_img = input_img.detach().cpu().numpy()
        inputs[0].set_data_from_numpy(input_img)
        output = self.triton_client.infer(model_name=self.model_name,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=1)
        outputs = [output.as_numpy('dets'), output.as_numpy('labels'), output.as_numpy('masks')]

        # outputs = output.as_numpy('masks')
        outputs = TensorrtDetector.__clear_outputs(outputs)
        return outputs
    def __call__(self, img, type_mask = "cpu"):
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
                    # mask = np.array(result[1][c][i]).astype(np.int)
                    mask = result[1][c][i].int()[None, ...]
                    roi = np.array(roi).astype(np.int)

                    # list_result.append([mask, roi, c, score])
                    classes.append(c)
                    scores.append(score)
                    boxes.append(roi)
                    masks.append(mask)
        masks = torch.cat(masks, dim = 0)
        if type_mask == "cpu":
            masks = masks.detach().cpu().numpy()
        return classes, scores, boxes, masks

    def plot(self, img):
        data = self.preprocess(img)
        outputs = self.predict(data)
        results = self.post_processing1(data, outputs)
        self.model.show_result(img, results[0], score_thr=0.3)

