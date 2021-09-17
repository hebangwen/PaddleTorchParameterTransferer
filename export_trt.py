import os
import cv2

import numpy as np
import paddle
from paddle.static import InputSpec

from pdparams.common.config import cfg
from pdparams.common.model import get_model
from pdparams.common.utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d
from pdparams.common.utils.vis import vis_keypoints, vis_3d_keypoints


if __name__ == "__main__":
    # 使用paddle.jit 将模型部署到TensorRT
    joint_num = 21 # single hand
    model = get_model("test", 21)
    model.load_dict(paddle.load('weights/interhand.pdparams'))
    model.eval()
    
    # out = model(inputs, targets, meta_info, 'test')
    image_shape = [3, 256, 256]
    input_spec = [InputSpec(shape=[None]+image_shape, name='inputs', dtype='float32')]

    # static_model = paddle.jit.to_static(model, input_spec=input_spec)
    paddle.jit.save(
        model,
        'output/interhand_trt/model',
        input_spec=input_spec
    )