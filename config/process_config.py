
from yacs.config import CfgNode as CN

_C = CN()

_C.SEGMETATION = CN()
_C.SEGMETATION.WEIGHTS = "./pretrain/yolact_base_last.pth"
_C.SEGMETATION.WEIGHTS_DAM = "./pretrain/yolact_base_damage_last.pth"
_C.SEGMETATION.MODEL_THRESH_PART = 0.8
_C.SEGMETATION.MODEL_THRESH_DAMAGE = 0.8
_C.SEGMETATION.FASTNMS = True
_C.SEGMETATION.CONFIG = "./swin_transformer/configs/insurance/config_swinT_cascade_2804.py"
_C.SEGMETATION.CHECKPOINT = "./pretrain/config_swinT_cascade_2804.pth"