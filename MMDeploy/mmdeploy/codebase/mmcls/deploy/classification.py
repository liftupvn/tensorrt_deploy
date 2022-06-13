# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset

from mmdeploy.codebase.base import BaseTask
from mmdeploy.utils import Task, get_root_logger
from mmdeploy.utils.config_utils import get_input_shape
from .mmclassification import MMCLS_TASK


def process_model_config(model_cfg: mmcv.Config,
                         imgs: Union[str, np.ndarray],
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (mmcv.Config): The model config.
        imgs (str | np.ndarray): Input image(s), accepted data type are `str`,
            `np.ndarray`.
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        mmcv.Config: the model config after processing.
    """
    cfg = model_cfg.deepcopy()
    if isinstance(imgs, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
    # check whether input_shape is valid
    if input_shape is not None:
        if 'crop_size' in cfg.data.test.pipeline[2]:
            crop_size = cfg.data.test.pipeline[2]['crop_size']
            if tuple(input_shape) != (crop_size, crop_size):
                logger = get_root_logger()
                logger.warning(
                    f'`input shape` should be equal to `crop_size`: {crop_size},\
                        but given: {input_shape}')
    return cfg


@MMCLS_TASK.register_module(Task.CLASSIFICATION.value)
class Classification(BaseTask):
    """Classification task class.

    Args:
        model_cfg (mmcv.Config): Original PyTorch model config file.
        deploy_cfg (mmcv.Config): Deployment config file or loaded Config
            object.
        device (str): A string represents device type.
    """

    def __init__(self, model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                 device: str):
        super(Classification, self).__init__(model_cfg, deploy_cfg, device)

    def init_backend_model(self,
                           model_files: Sequence[str] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .classification_model import build_classification_model

        model = build_classification_model(
            model_files, self.model_cfg, self.deploy_cfg, device=self.device)

        return model.eval()

    def init_pytorch_model(self,
                           model_checkpoint: Optional[str] = None,
                           cfg_options: Optional[Dict] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize torch model.

        Args:
            model_checkpoint (str): The checkpoint file of torch model,
                Default: None.
            cfg_options (dict): Optional config key-pair parameters.

        Returns:
            nn.Module: An initialized torch model generated by OpenMMLab
                codebases.
        """
        from mmcls.apis import init_model
        model = init_model(self.model_cfg, model_checkpoint, self.device,
                           cfg_options)

        return model.eval()

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Optional[Sequence[int]] = None) \
            -> Tuple[Dict, torch.Tensor]:
        """Create input for classifier.

        Args:
            imgs (Any): Input image(s), accepted data type are `str`,
                `np.ndarray`, `torch.Tensor`.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Default: None.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        from mmcls.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        cfg = process_model_config(self.model_cfg, imgs, input_shape)
        if isinstance(imgs, str):
            data = dict(img_info=dict(filename=imgs), img_prefix=None)
        else:
            data = dict(img=imgs)
        test_pipeline = Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data['img'] = [data['img']]
        if self.device != 'cpu':
            data = scatter(data, [self.device])[0]
        return data, data['img']

    def visualize(self,
                  model: torch.nn.Module,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False):
        """Visualize predictions of a model.

        Args:
            model (nn.Module): Input model.
            image (str | np.ndarray): Input image to draw predictions on.
            result (list): A list of predictions.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows.
                Default: False.
        """
        show_img = mmcv.imread(image) if isinstance(image, str) else image
        output_file = None if show_result else output_file
        pred_score = np.max(result)
        pred_label = np.argmax(result)
        result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
        result['pred_class'] = model.CLASSES[result['pred_label']]
        return model.show_result(
            show_img,
            result,
            show=show_result,
            win_name=window_name,
            out_file=output_file)

    @staticmethod
    def run_inference(model: torch.nn.Module,
                      model_inputs: Dict[str, torch.Tensor]) -> list:
        """Run inference once for a classification model of mmcls.

        Args:
            model (nn.Module): Input model.
            model_inputs (dict): A dict containing model inputs tensor and
                meta info.

        Returns:
            list: The predictions of model inference.
        """
        return model(**model_inputs, return_loss=False)

    @staticmethod
    def get_partition_cfg(partition_type: str) -> Dict:
        """Get a certain partition config.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        raise NotImplementedError('Not supported yet.')

    @staticmethod
    def get_tensor_from_input(input_data: Dict[str, Any]) -> torch.Tensor:
        """Get input tensor from input data.

        Args:
            input_data (tuple): Input data containing meta info and image
            tensor.
        Returns:
            torch.Tensor: An image in `Tensor`.
        """
        return input_data['img']

    @staticmethod
    def evaluate_outputs(model_cfg: mmcv.Config,
                         outputs: list,
                         dataset: Dataset,
                         metrics: Optional[str] = None,
                         out: Optional[str] = None,
                         metric_options: Optional[dict] = None,
                         format_only: bool = False,
                         log_file: Optional[str] = None) -> None:
        """Perform post-processing to predictions of model.

        Args:
            model_cfg (mmcv.Config): The model config.
            outputs (list): A list of predictions of model inference.
            dataset (Dataset): Input dataset to run test.
            metrics (str): Evaluation metrics, which depends on
                the codebase and the dataset, e.g., "mAP" in mmcls.
            out (str): Output result file in pickle format, Default: None.
            metric_options (dict): Custom options for evaluation, will be
                kwargs for dataset.evaluate() function. Default: None.
            format_only (bool): Format the output results without perform
                evaluation. It is useful when you want to format the result
                to a specific format and submit it to the test server.
                Default: False.
            log_file (str | None): The file to write the evaluation results.
                Defaults to `None` and the results will only print on stdout.
        """
        import warnings

        from mmcv.utils import get_logger
        logger = get_logger('test', log_file=log_file, log_level=logging.INFO)

        if metrics:
            results = dataset.evaluate(outputs, metrics, metric_options)
            for k, v in results.items():
                logger.info(f'{k} : {v:.2f}')
        else:
            warnings.warn('Evaluation metrics are not specified.')
            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [dataset.CLASSES[lb] for lb in pred_label]
            results = {
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_class': pred_class
            }
            if not out:
                logger.info('the predicted result for the first element is '
                            f'pred_score = {pred_score[0]:.2f}, '
                            f'pred_label = {pred_label[0]} '
                            f'and pred_class = {pred_class[0]}. '
                            'Specify --out to save all results to files.')
        if out:
            logger.debug(f'writing results to {out}')
            mmcv.dump(results, out)

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        cfg = process_model_config(self.model_cfg, '', input_shape)
        preprocess = cfg.data.test.pipeline
        return preprocess

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        postprocess = self.model_cfg.model.head
        assert 'topk' in postprocess, 'model config lack topk'
        postprocess.topk = max(postprocess.topk)
        return postprocess

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'backbone' in self.model_cfg.model, 'backbone not in model '
        'config'
        assert 'type' in self.model_cfg.model.backbone, 'backbone contains '
        'no type'
        name = self.model_cfg.model.backbone.type.lower()
        return name