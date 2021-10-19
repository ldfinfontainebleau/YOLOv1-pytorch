#!/usr/bin/env python3
"""
@author: LiuDanfeng
@since: 2021-09-30
"""
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, resnet50
import torch.nn.functional as F
# import sys
# sys.path.append("..")
import config


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, leaky=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if leaky else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


# Copy from yolov5
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_, c2, k=3, p=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = nn.Conv2d(c1, c_, kernel_size=1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, kernel_size=1, bias=False)
        self.cv4 = Conv(2 * c_, c2, k=1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SAM(nn.Module):
    """ Parallel CBAM """
    def __init__(self, in_ch):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Spatial Attention Module """
        x_attention = self.conv(x)

        return x * x_attention


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x


class DetnetBottleNeck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(DetnetBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class YOLOv1(nn.Module):

    def __init__(self):
        super(YOLOv1, self).__init__()
        backbone = []
        if config.backbone == 'resnet18':
            for child in resnet18(pretrained=config.pretrained).children():
                if isinstance(child, nn.AdaptiveAvgPool2d):
                    break
                if config.freeze_backbone:
                    child.eval()
                    for param in child.parameters():
                        param.requires_grad = False
                backbone.append(child)
            self.backbone = nn.Sequential(*backbone)
        elif config.backbone == 'resnet50':
            for child in resnet50(pretrained=config.pretrained).children():
                if isinstance(child, nn.AdaptiveAvgPool2d):
                    break
                if config.freeze_backbone:
                    for param in child.parameters():
                        param.requires_grad = False
                backbone.append(child)
            self.backbone = nn.Sequential(*backbone)
            dummpy = torch.FloatTensor(1, 3, 448, 448)
            dummpy = self.backbone(dummpy)
            print(dummpy.shape)
            exit()
        self.head = YOLOv1Head()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = x.permute((0, 2, 3, 1))
        return x


class YOLOv1Head(nn.Module):

    def __init__(self):
        super(YOLOv1Head, self).__init__()
        if config.backbone == 'resnet18':
            if config.head == 'normal':
                self.net = nn.Sequential(
                    nn.Conv2d(512, 1024, (3, 3), (2, 2), (1, 1)),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(1024, 5 * config.boxes_per_grid + config.classes, (3, 3), (1, 1), (1, 1)),
                    nn.Sigmoid(),
                )
            elif config.head == 'long':
                self.net = nn.Sequential(
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0)),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1)),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0)),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(1024, 5 * config.boxes_per_grid + config.classes, (3, 3), (1, 1), (1, 1)),
                    nn.Sigmoid(),
                )
            elif config.head == 'detnet':
                self.net = nn.Sequential(
                    DetnetBottleNeck(in_planes=512, planes=256, block_type='B'),
                    DetnetBottleNeck(in_planes=256, planes=256, block_type='A'),
                    DetnetBottleNeck(in_planes=256, planes=256, block_type='A'),
                    nn.Conv2d(256, 5 * config.boxes_per_grid + config.classes, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(5 * config.boxes_per_grid + config.classes),
                    nn.Sigmoid()
                )
            elif config.head == 'v3':
                self.net = nn.Sequential(
                    Conv(512, 256, k=3, s=2, p=1),
                    SPP(),
                    BottleneckCSP(256 * 4, 512, n=1, shortcut=False),
                    SAM(512),
                    BottleneckCSP(512, 512, n=3, shortcut=False),
                    nn.Conv2d(512, 5 * config.boxes_per_grid + config.classes, 1),
                    nn.Sigmoid()
                )
        elif config.backbone == 'resnet50':
            self.net = nn.Sequential(
                nn.Conv2d(2048, 1024, (3, 3), (2, 2), (1, 1)),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(1024, 5 * config.boxes_per_grid + config.classes, (3, 3), (1, 1), (1, 1)),
                nn.BatchNorm2d(5 * config.boxes_per_grid + config.classes),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x = self.net(x)
        return x


class Decoder(nn.Module):
    # TODO(liudf6): use numpy mostly
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, pred: torch.Tensor):
        """Turn YOLOv1 net output feature into bboxes
        Args:
            pred: Tensor (b, grids, grids, boxes+classes) [cx, cy, w, h]
        Returns:
            bboxes: List of Numpy.array [b, (n, 6)] [x1, y1, x2, y2, cls, prob]
        """
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        batch = pred.shape[0]
        grid = pred.shape[1]
        contain_prob = pred[..., [4, 9]]  # (b, g, g, 2)
        max_cls_prob = np.max(pred[..., 10:], axis=3)  # (b, g, g)
        cls_index = np.argmax(pred[..., 10:], axis=3)  # (b, g, g)
        cls_index = np.expand_dims(cls_index, axis=3).repeat(2, axis=3)  # (b, g, g, 2)
        probs = contain_prob * np.expand_dims(max_cls_prob, axis=3).repeat(2, axis=3)  # (b, g, g, 2)
        probs_mask = np.where(probs > config.obj_threshold, 1., 0.)  # (b, g, g, 2)
        box_pred = pred[..., [0, 1, 2, 3, 5, 6, 7, 8]].reshape((batch, grid, grid, 2, -1))  # (b, g, g, 2, 4)
        # # grid index for coord tranfer by lvgy.
        # y, x = np.meshgrid(np.arange(grid), np.arange(grid))  # (g, g)
        # xy_grid_index = np.expand_dims(np.stack([x, y], 2), axis=0).repeat(batch, axis=0)  # (b, g, g, 2)
        # # xy_grid_index = np.expand_dims(np.stack([y, x], 2), axis=0).repeat(batch, axis=0)  # (b, g, g, 2)
        #
        # xy_grid_index = np.expand_dims(xy_grid_index, axis=4).repeat(2, axis=4)  # (b, g, g, 2, 2)
        # # coord transfer. grid to pic.
        # xy_x = (box_pred[..., 0] + xy_grid_index[..., 0]) / grid  # (b, g, g, 2)
        # xy_y = (box_pred[..., 1] + xy_grid_index[..., 1]) / grid  # (b, g, g, 2)
        # grid index for coord tranfer.
        x_grid_index = np.expand_dims(np.arange(0, grid, 1), axis=0).repeat(grid, axis=0)  # (g, g)
        x_grid_index = np.expand_dims(x_grid_index, axis=(0, 3)).repeat(batch, axis=0).repeat(2, axis=3)  # (b, g, g, 2)
        y_grid_index = np.expand_dims(np.arange(0, grid, 1), axis=1).repeat(grid, axis=1)  # (g, g)
        y_grid_index = np.expand_dims(y_grid_index, axis=(0, 3)).repeat(batch, axis=0).repeat(2, axis=3)  # (b, g, g, 2)
        # coord transfer. grid to pic.
        xy_x = (box_pred[..., 0] + x_grid_index) / grid  # (b, g, g, 2)
        xy_y = (box_pred[..., 1] + y_grid_index) / grid  # (b, g, g, 2)
        x1 = np.expand_dims(xy_x - 0.5*box_pred[..., 2], axis=4)  # (b, g, g, 2, 1)
        y1 = np.expand_dims(xy_y - 0.5*box_pred[..., 3], axis=4)
        x2 = np.expand_dims(xy_x + 0.5*box_pred[..., 2], axis=4)
        y2 = np.expand_dims(xy_y + 0.5*box_pred[..., 3], axis=4)
        x1[x1 < 0] = 0
        y1[y1 < 0] = 0
        x2[x2 > 1] = 1
        y2[y2 > 1] = 1
        decode_pred = np.concatenate((x1, y1, x2, y2), axis=4)  # (b, g, g, 2, 4)
        decode_pred = np.concatenate((
            decode_pred,
            np.expand_dims(cls_index, axis=4),
            np.expand_dims(probs, axis=4)
        ), axis=4)  # (b, g, g, 2, 6)
        decode_pred = decode_pred*np.expand_dims(probs_mask, axis=4).repeat(6, axis=4)  # (b, g, g, 2, 6)
        decode_pred = decode_pred.reshape((batch, -1, 6))  # (b, g*g*2, 6)
        bboxes = self._nms(decode_pred)
        return bboxes

    def _nms(self, pred_boxes):
        """Filter other boxes
        Args:
            pred_boxes: (b, n, 6) numpy array, [x1, y1, x2, y2, cls, prob].
        Returns:
            bboxes: b x [m, 6] lists of numpy array, m is not a fix num.
        """
        bboxes = list()
        for i in range(pred_boxes.shape[0]):
            # shrink zeros
            bbox = pred_boxes[i, ]  # (n, 6)
            index = np.argwhere(np.all(bbox[:, ] == 0., axis=1))
            bbox = np.delete(bbox, index, axis=0)  # (m, 6)
            if len(bbox) == 0:
                bboxes.append([])
                continue
            bbox = bbox[np.argsort(-bbox[:, 5]), :]  # (m, 6)
            keep_box = list()
            while len(bbox) > 0:
                if len(bbox) == 1:
                    keep_box.append(bbox[0, :])
                    break
                master_box = np.expand_dims(bbox[0, :4], axis=0)  # (4,) -> (1, 4)
                salve_box = bbox[1:, :4]  # (k, 4)
                inter_x1y1 = np.maximum(master_box[..., :2], salve_box[..., :2])  # (k, 2)
                inter_x2y2 = np.minimum(master_box[..., 2:], salve_box[..., 2:])  # (k, 2)
                inter_wh = np.clip((inter_x2y2 - inter_x1y1), 0, None)  # (k, 2)
                inter_area = inter_wh[:, 0] * inter_wh[:, 1]  # (k,)
                master_area = (master_box[:, 2] - master_box[:, 0]) * (master_box[:, 3] - master_box[:, 1])  # (1,)
                slave_area = (salve_box[:, 2] - salve_box[:, 0]) * (salve_box[:, 3] - salve_box[:, 1])  # (k,)
                union_area = master_area + slave_area - inter_area  # (k,)
                ious = inter_area / union_area  # (k,)
                del_index = np.argwhere(ious > config.nms_threshold)
                keep_box.append(bbox[0, :])
                bbox = np.delete(bbox, 0, axis=0)
                bbox = np.delete(bbox, del_index, axis=0)
            bboxes.append(np.array(keep_box))
        return bboxes


class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()

    def forward(self, pred: torch.Tensor):
        """Turn YOLOv1 net output feature into bboxes
        Args:
            pred: Tensor (b, grids, grids, 5+classes) [cx, cy, w, h, contain, cls...]
        Returns:
            bboxes: List of Numpy.array [b, (n, 6)] [x1, y1, x2, y2, cls, prob]
        """
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        batch = pred.shape[0]
        grid = pred.shape[1]
        contain_prob = pred[..., 4]  # (b, g, g)
        max_cls_prob = np.max(pred[..., 5:], axis=3)  # (b, g, g)
        cls_index = np.argmax(pred[..., 5:], axis=3)  # (b, g, g)
        probs = contain_prob * max_cls_prob  # (b, g, g)
        probs_mask = np.where(probs > config.obj_threshold, 1., 0.)  # (b, g, g)
        box_pred = pred[..., :4]  # (b, g, g, 4)
        # grid index for coord tranfer by lvgy.
        y, x = np.meshgrid(np.arange(grid), np.arange(grid))  # (g, g)
        xy_grid_index = np.expand_dims(np.stack([x, y], 2), axis=0).repeat(batch, axis=0)  # (b, g, g, 2)
        # coord transfer. grid to pic.
        xy_x = (box_pred[..., 0] + xy_grid_index[..., 0]) / grid  # (b, g, g)
        xy_y = (box_pred[..., 1] + xy_grid_index[..., 1]) / grid  # (b, g, g)
        x1 = np.expand_dims(xy_x - 0.5*box_pred[..., 2], axis=3)  # (b, g, g, 1)
        y1 = np.expand_dims(xy_y - 0.5*box_pred[..., 3], axis=3)
        x2 = np.expand_dims(xy_x + 0.5*box_pred[..., 2], axis=3)
        y2 = np.expand_dims(xy_y + 0.5*box_pred[..., 3], axis=3)
        decode_pred = np.concatenate((x1, y1, x2, y2), axis=3)  # (b, g, g, 4)
        decode_pred = np.concatenate((
            decode_pred,
            np.expand_dims(cls_index, axis=3),
            np.expand_dims(probs, axis=3)
        ), axis=3)  # (b, g, g, 6)
        decode_pred = decode_pred*np.expand_dims(probs_mask, axis=3).repeat(6, axis=3)  # (b, g, g, 6)
        decode_pred = decode_pred.reshape((batch, -1, 6))  # (b, g*g, 6)
        bboxes = self._nms(decode_pred)
        return bboxes

    def _nms(self, pred_boxes):
        """Filter other boxes
        Args:
            pred_boxes: (b, n, 6) numpy array, [x1, y1, x2, y2, cls, prob].
        Returns:
            bboxes: b x [m, 6] lists of numpy array, m is not a fix num.
        """
        bboxes = list()
        for i in range(pred_boxes.shape[0]):
            # shrink zeros
            bbox = pred_boxes[i, ]  # (n, 6)
            index = np.argwhere(np.all(bbox[:, ] == 0., axis=1))
            bbox = np.delete(bbox, index, axis=0)  # (m, 6)
            if len(bbox) == 0:
                bboxes.append([])
                continue
            bbox = bbox[np.argsort(-bbox[:, 5]), :]  # (m, 6)
            keep_box = list()
            while len(bbox) > 0:
                if len(bbox) == 1:
                    keep_box.append(bbox[0, :])
                    break
                master_box = np.expand_dims(bbox[0, :4], axis=0)  # (4,) -> (1, 4)
                salve_box = bbox[1:, :4]  # (k, 4)
                inter_x1y1 = np.maximum(master_box[..., :2], salve_box[..., :2])  # (k, 2)
                inter_x2y2 = np.minimum(master_box[..., 2:], salve_box[..., 2:])  # (k, 2)
                inter_wh = np.clip((inter_x2y2 - inter_x1y1), 0, None)  # (k, 2)
                inter_area = inter_wh[:, 0] * inter_wh[:, 1]  # (k,)
                master_area = (master_box[:, 2] - master_box[:, 0]) * (master_box[:, 3] - master_box[:, 1])  # (1,)
                slave_area = (salve_box[:, 2] - salve_box[:, 0]) * (salve_box[:, 3] - salve_box[:, 1])  # (k,)
                union_area = master_area + slave_area - inter_area  # (k,)
                ious = inter_area / union_area  # (k,)
                del_index = np.argwhere(ious > config.nms_threshold)
                keep_box.append(bbox[0, :])
                bbox = np.delete(bbox, 0, axis=0)
                bbox = np.delete(bbox, del_index, axis=0)
            bboxes.append(np.array(keep_box))
        return bboxes


def test():
    decoder = Decoder()
    dummy = np.load('test_feature.npz')
    # real data
    b = decoder(torch.tensor(dummy['arr_0']))
    b[0][:, :4] = b[0][:, :4]*448
    print(b)
    # plot it
    import cv2
    image = cv2.imread('test.jpg')
    for i in range(len(b[0])):
        cv2.rectangle(image, (int(b[0][i, 0]), int(b[0][i, 1])), (int(b[0][i, 2]), int(b[0][i, 3])), (30, 105, 210), 1)
    cv2.imwrite('test_result.jpg', image)

    return 0


if __name__ == '__main__':
    test()
