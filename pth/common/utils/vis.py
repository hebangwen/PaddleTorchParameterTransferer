# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import cv2
import numpy
import numpy as np
import matplotlib
matplotlib.use('tkagg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from pth.common.config import cfg
from PIL import Image, ImageDraw

def get_keypoint_rgb(skeleton):
    rgb_dict= {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        
    return rgb_dict


def get_keypoint_rgb2(skeleton):
    rgb_dict = {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        # (255, 51, 51)
        # (51, 153, 255)

        if joint_name.startswith('l'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.startswith('r'):
            rgb_dict[joint_name] = (51, 153, 255)

    return rgb_dict


def vis_keypoints(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path=None):
    rgb_dict = get_keypoint_rgb2(skeleton)
    _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
        if score[i] > score_thr:
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
    
    if save_path is None:
        _img.save(osp.join(cfg.vis_dir, filename))
    else:
        _img.save(osp.join(save_path, filename))

    return np.array(_img)


def vis_3d_keypoints(kps_3d, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb2(skeleton)
    
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i,0], kps_3d[pid,0]])
        y = np.array([kps_3d[i,1], kps_3d[pid,1]])
        z = np.array([kps_3d[i,2], kps_3d[pid,2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')
    
    fig.savefig(osp.join(cfg.vis_dir, filename), dpi=fig.dpi)
    plt.close(fig)


def vis_3d_keypoints_and_2d_image(kps_3d, skeleton, bbox, vis_2d_image, line_width=3):
    # vis 3d and 2d
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    rgb_dict = get_keypoint_rgb2(skeleton)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i, 0], kps_3d[pid, 0]])
        y = np.array([kps_3d[i, 1], kps_3d[pid, 1]])
        z = np.array([kps_3d[i, 2], kps_3d[pid, 2]])

        ax.plot(x, z, -y, c=np.array(rgb_dict[parent_joint_name]) / 255., linewidth=line_width)
        ax.scatter(kps_3d[i, 0], kps_3d[i, 2], -kps_3d[i, 1], c=np.array(rgb_dict[joint_name]).reshape(1, 3) / 255., marker='o')
        ax.scatter(kps_3d[pid, 0], kps_3d[pid, 2], -kps_3d[pid, 1], c=np.array(rgb_dict[parent_joint_name]).reshape(1, 3) / 255., marker='o')

    xmax, xmin = np.max(kps_3d[:, 0]), np.min(kps_3d[:, 0])
    ymax, ymin = np.max(kps_3d[:, 1]), np.min(kps_3d[:, 1])
    zmax, zmin = np.max(kps_3d[:, 2]), np.min(kps_3d[:, 2])

    # 裁剪bbox内容
    vis_img = vis_2d_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

    # 以x和y为中心点, 扩展w和h
    xhat = (xmin + xmax) / 2
    yhat = (ymin + ymax) / 2

    y, x = np.ogrid[-yhat-bbox[3]/2:-yhat+bbox[3]/2, xhat-bbox[2]/2:xhat+bbox[2]/2]
    vis_img = cv2.rotate(vis_img, cv2.cv2.ROTATE_180)
    vis_img = vis_img[:, ::-1, :]
    vis_img = vis_img.astype(np.float32)
    vis_img /= 255

    ax.plot_surface(x, np.atleast_2d(zmax + 100), y, rstride=1, cstride=1, facecolors=vis_img)

    plt.show()
