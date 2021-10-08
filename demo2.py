# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

from pth.common.config import cfg
from pth.common.model import get_model
from pth.common.utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d
from pth.common.utils.vis import vis_keypoints, vis_3d_keypoints, vis_3d_keypoints_and_2d_image
from pth.common.utils.transforms import pixel2cam


if __name__ == '__main__':
    joint_num = 21 # single hand
    root_joint_idx = {'right': 20, 'left': 41}
    joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}
    skeleton = load_skeleton(osp.join('annotations/skeleton.txt'), joint_num*2)

    # snapshot load
    model_path = 'weights/snapshot_20.pth.tar'
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_model('test', joint_num)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = model.state_dict()

    for k in ckpt['network'].keys():
        t = k.replace('module.', '')
        state_dict[t] = ckpt['network'][k]

    model.cpu()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # prepare input image
    transform = transforms.ToTensor()
    img_path = 'images/input.jpg'
    original_img = cv2.imread(img_path)
    original_img_height, original_img_width = original_img.shape[:2]

    # prepare bbox
    bbox = [69, 137, 165, 153] # (input.jpg) xmin, ymin, width, height
    pbbox = process_bbox(bbox.copy(), (original_img_height, original_img_width, original_img_height))
    img, trans, inv_trans = generate_patch_image(original_img, pbbox, False, 1.0, 0.0, cfg.input_img_shape)
    img = transform(img.astype(np.float32))/255
    img = img[None,:,:,:]

    # forward
    inputs = {'img': img}
    targets = {}
    meta_info = {}
    out = model(inputs, targets, meta_info, 'test')
    img = img[0].cpu().numpy().transpose(1,2,0)
    joint_coord = out['joint_coord'][0].detach().cpu().numpy()
    rel_root_depth = out['rel_root_depth'][0].detach().cpu().numpy()
    hand_type = out['hand_type'][0].detach().cpu().numpy()

    # restore joint coord to original image space and continuous depth space
    joint_coord[:,0] = joint_coord[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    joint_coord[:,1] = joint_coord[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    joint_coord[:,:2] = np.dot(inv_trans, np.concatenate((joint_coord[:,:2], np.ones_like(joint_coord[:,:1])),1).transpose(1,0)).transpose(1,0)
    joint_coord[:,2] = (joint_coord[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

    # restore right hand-relative left hand depth to continuous depth space
    rel_root_depth = (rel_root_depth/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

    # right hand root depth == 0, left hand root depth == rel_root_depth
    joint_coord[joint_type['left'],2] += rel_root_depth

    # handedness
    joint_valid = np.zeros((joint_num*2), dtype=np.float32)
    right_exist = False
    if hand_type[0] > 0.5:
        right_exist = True
        joint_valid[joint_type['right']] = 1
    left_exist = False
    if hand_type[1] > 0.5:
        left_exist = True
        joint_valid[joint_type['left']] = 1

    print('Right hand exist: ' + str(right_exist) + ' Left hand exist: ' + str(left_exist))
    np.save("demo2_coord.npy", joint_coord)

    # visualize joint coord in 2D space
    filename = 'images/result_2d.jpg'
    vis_img = original_img.copy()[:,:,::-1].transpose(2,0,1)
    vis_img = vis_keypoints(vis_img, joint_coord, joint_valid, skeleton, filename, save_path='.')

    filename = 'result_3d'
    vis_3d_keypoints(joint_coord, joint_valid, skeleton, filename)

    vis_3d_keypoints_and_2d_image(joint_coord, skeleton, bbox, vis_img)