import os
import numpy as np
import cv2
import paddle
import yaml

from paddle.inference import Config
from paddle.inference import create_predictor

import paddle.vision.transforms as _transforms
from pdparams.common.model import get_model


SUPPORT_MODELS = ['YOLO']


class PredictConfig:
    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)

        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.input_shape = yml_conf['image_shape']


# You can only start inference after putting input tensor to `input handle`
# If you want to get rid of this, you need to change the export mode.
class Detector(object):
    """
    Args:
        config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        use_gpu (bool): whether use gpu
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        threshold (float): threshold to reserve the result for output.
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 use_dynamic_shape=False,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 threshold=0.5):
        self.pred_config = pred_config
        self.predictor = self.load_predictor(
            model_dir,
            run_mode=run_mode,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            use_gpu=use_gpu,
            use_dynamic_shape=use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape)
        self.threshold = threshold

    def load_predictor(self,
                       model_dir,
                       run_mode='fluid',
                       batch_size=1,
                       use_gpu=False,
                       min_subgraph_size=3,
                       use_dynamic_shape=False,
                       trt_min_shape=1,
                       trt_max_shape=1280,
                       trt_opt_shape=640):
        """set AnalysisConfig, generate AnalysisPredictor
        Args:
            model_dir (str): root path of __model__ and __params__
            use_gpu (bool): whether use gpu
            run_mode (str): mode of running(fluid/trt_fp32/trt_fp16/trt_int8)
            use_dynamic_shape (bool): use dynamic shape or not
            trt_min_shape (int): min shape for dynamic shape in trt
            trt_max_shape (int): max shape for dynamic shape in trt
            trt_opt_shape (int): opt shape for dynamic shape in trt
        Returns:
            predictor (PaddlePredictor): AnalysisPredictor
        Raises:
            ValueError: predict by TensorRT need use_gpu == True.
        """
        if not use_gpu and not run_mode == 'fluid':
            raise ValueError(
                "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
                    .format(run_mode, use_gpu))
        use_calib_mode = True if run_mode == 'trt_int8' else False
        config = Config(
            os.path.join(model_dir, 'model.pdmodel'),
            os.path.join(model_dir, 'model.pdiparams'))
        precision_map = {
            'trt_int8': Config.Precision.Int8,
            'trt_fp32': Config.Precision.Float32,
            'trt_fp16': Config.Precision.Half
        }
        if use_gpu:
            # initial GPU memory(M), device ID
            config.enable_use_gpu(200, 0)
            # optimize graph and fuse op
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()

        if run_mode in precision_map.keys():
            config.enable_tensorrt_engine(
                workspace_size=1 << 10,
                max_batch_size=batch_size,
                min_subgraph_size=min_subgraph_size,
                precision_mode=precision_map[run_mode],
                use_static=False,
                use_calib_mode=use_calib_mode)

            if use_dynamic_shape:
                print('use_dynamic_shape')
                min_input_shape = {'image': [1, 3, trt_min_shape, trt_min_shape]}
                max_input_shape = {'image': [1, 3, trt_max_shape, trt_max_shape]}
                opt_input_shape = {'image': [1, 3, trt_opt_shape, trt_opt_shape]}
                config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                                  opt_input_shape)
                print('trt set dynamic shape done!')

        # disable print log when predict
        config.disable_glog_info()
        # enable shared memory
        config.enable_memory_optim()
        # disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)
        return predictor

    def postprocess(self, np_boxes, inputs):
        # postprocess output of predictor
        if self.pred_config.arch in ['Face']:
            h, w = inputs['im_shape']
            scale_y, scale_x = inputs['scale_factor']
            w, h = float(h) / scale_y, float(w) / scale_x
            np_boxes[:, 2] *= h
            np_boxes[:, 3] *= w
            np_boxes[:, 4] *= h
            np_boxes[:, 5] *= w
        results = (np_boxes[:, 1] > self.threshold) & (np_boxes[:, 0] > -1)
        return results

    def predict(self, inputs, warmup=0, repeats=1):
        """
        Args:
            inputs: dict, keys: `image`, `im_shape`, `scale_factor`
            threshold: bbox threshold
            warmup: model warmup rounds
            repeats: model repeat round

        Returns:
            result: boxes filtered by threshold, shape: Nx6, [class, score, x_min, y_min, x_max, y_max]
        """
        np_boxes, np_masks = None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()

        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()

        return self.postprocess(np_boxes, inputs)


def generate_input(image_path, cfg: PredictConfig):
    image_src = cv2.imread(image_path)
    image_size = np.array(image_src.shape[:-1])
    target_size = np.array(cfg.input_shape[1:])

    transform = _transforms.Compose([
        _transforms.Resize(tuple(target_size)),
        _transforms.ToTensor(),
        _transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    scale_factor = target_size / image_size
    image_tensor = transform(image_src)
    image_tensor = paddle.unsqueeze(image_tensor, axis=0)
    inputs = {'image': image_tensor, 'im_shape': target_size, 'scale_factor': scale_factor}

    return image_src, inputs


if __name__ == "__main__":
    model_dir = "ppyolo_r18vd_coco"
    cfg = PredictConfig(model_dir)
    img_src, inputs = generate_input("demo/256.jpg", cfg)

    detector = Detector(
        cfg,
        model_dir,
        run_mode="trt_fp32",
        threshold=0.45
    )

    result = detector.predict(inputs, warmup=3)
    print(result)