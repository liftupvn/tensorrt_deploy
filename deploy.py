from MMDeploy.infer2 import TensorrtDetector

import cv2
img = cv2.imread("MMDeploy/demo/demo.jpg")
img = cv2.resize(img, (800, 800))
deploy_cfg_path = "MMDeploy/configs/mmdet/instance-seg/instance-seg_tensorrt-int8_dynamic-320x320-1344x1344.py"
model_cfg_path = "mmdetection/configs/insurance/cascade_mask_rcnn_restnext101.py"
model_file = ['MMDeploy/model_convert/end2end.engine']
detector = TensorrtDetector(deploy_cfg_path, model_cfg_path, model_file)
print(detector(img))
