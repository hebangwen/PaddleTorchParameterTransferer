from dnns.internet import config
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class JointHeatmapLoss(nn.Layer):
    def __ini__(self):
        super(JointHeatmapLoss, self).__init__()

    def forward(self, joint_out, joint_gt, joint_valid):
        # loss = (joint_out - joint_gt) ** 2 * joint_valid[:, :, None, None, None]
        loss = (joint_out - joint_gt) ** 2 * paddle.unsqueeze(joint_valid, axis=[-1, -1, -1])
        return loss


class HandTypeLoss(nn.Layer):
    def __init__(self):
        super(HandTypeLoss, self).__init__()

    def forward(self, hand_type_out, hand_type_gt, hand_type_valid):
        loss = F.binary_cross_entropy(hand_type_out, hand_type_gt, reduction='none')
        loss = loss.mean(1)
        loss = loss * hand_type_valid

        return loss


class RelRootDepthLoss(nn.Layer):
    def __init__(self):
        super(RelRootDepthLoss, self).__init__()

    def forward(self, root_depth_out, root_depth_gt, root_valid):
        loss = paddle.abs(root_depth_out - root_depth_gt) * root_valid
        return loss
