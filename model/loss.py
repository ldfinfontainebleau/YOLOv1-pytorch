#!/usr/bin/env python3
"""
@author: LiuDanfeng
@since: 2021-10-08
"""
import torch
from torch import nn
from torch.nn import functional as F
# import sys
# sys.path.append("..")
import config


class YOLOv1Loss(nn.Module):

    def __init__(self):
        super(YOLOv1Loss, self).__init__()

    def forward(self, pred, bboxes):
        """Compute loss with YOLOv1 interdoction
        Args:
            pred: (batch, grid, grid, boxes+classes)
            bboxes: List of array [b x (n, 5)]
        Return:
            loss: tensor float
        """
        device = pred.device
        target = self._make_target(device, bboxes)
        obj_mask = target[:, :, :, [4, 9]]  # (n, g, g, 2)
        noobj_mask = (1.0 - obj_mask) * config.lamda_noobj  # (n, g, g, 2)
        loss_contain = (target[:, :, :, [4, 9]] - pred[:, :, :, [4, 9]]) ** 2  # (n, g, g, 2)
        loss_contain = (obj_mask + noobj_mask) * loss_contain  # (n, g, g, 2)
        loss_contain = loss_contain.sum((1, 2, 3))  # (n,)
        loss_coord = (
                (target[:, :, :, [0, 5]] - pred[:, :, :, [0, 5]]) ** 2 +
                (target[:, :, :, [1, 6]] - pred[:, :, :, [1, 6]]) ** 2 +
                (target[:, :, :, [2, 7]].sqrt() - pred[:, :, :, [2, 7]].sqrt()) ** 2 +
                (target[:, :, :, [3, 8]].sqrt() - pred[:, :, :, [3, 8]].sqrt()) ** 2
        )  # (n, g, g, 2)
        loss_coord = obj_mask * loss_coord * config.lamda_coor  # (n, g, g)
        loss_coord = loss_coord.sum((1, 2, 3))
        loss_class = (target[:, :, :, 10:] - pred[:, :, :, 10:]) ** 2
        loss_class = obj_mask.max(3, keepdim=True).values * loss_class
        loss_class = loss_class.sum((1, 2, 3))
        return (loss_contain + loss_coord + loss_class).mean()

    def _make_target(self, device, bboxes_list):
        """make bbox to feature like
        Args:
            device: cpu or cuda index
            bboxes_list: List of array [b x (n, 5)]
        Return:
            target: (batch, grid, grid, boxes+classes)
        """
        # assert config.batch_size == len(bboxes_list)
        if config.batch_size != len(bboxes_list):
            batch_size = len(bboxes_list)
        else:
            batch_size = config.batch_size
        target = torch.zeros(
            (batch_size, config.grids, config.grids, 5*config.boxes_per_grid + config.classes),
            dtype=torch.float32,
            device=device
        )
        image_size = torch.FloatTensor(
            [config.image_size[0], config.image_size[1], config.image_size[0], config.image_size[1]]
        ).to(device)
        for i, bboxes in enumerate(bboxes_list):
            # bboxes: (?, 5)
            box = bboxes[:, :4] / image_size  # (?, 4) float32
            label = bboxes[:, 4].long()  # (?,) int64
            x_center = (box[:, 0] + box[:, 2]) * 0.5  # all pic
            y_center = (box[:, 1] + box[:, 3]) * 0.5
            w_box = box[:, 2] - box[:, 0]
            h_box = box[:, 3] - box[:, 1]
            class_target = F.one_hot(label, config.classes).float()  # (n, 20)
            x_grid = (x_center * config.grids).long()  # which grid
            y_grid = (y_center * config.grids).long()
            x_offset = x_center * config.grids - x_grid
            y_offset = y_center * config.grids - y_grid
            for xg, yg, xo, yo, w, h, clss in zip(x_grid, y_grid, x_offset, y_offset, w_box, h_box, class_target):

                # big first, small second.
                area = w * h
                if area > config.obj_area_threshold:
                    if target[i, yg, xg, 4] == 0:
                        target[i, yg, xg, :5] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                        if target[i, yg, xg, 9] == 1.0 and not clss.equal(target[i, yg, xg, 10:]):
                            # remove small and not same class obj.
                            target[i, yg, xg, 5:10] = torch.FloatTensor([0, 0, 0, 0, 0]).to(device)
                            target[i, yg, xg, 10:] = clss
                        elif target[i, yg, xg, 9] == 0:
                            target[i, yg, xg, 10:] = clss
                else:
                    if target[i, yg, xg, 9] == 0:
                        if target[i, yg, xg, 4] == 1.0 and clss.equal(target[i, yg, xg, 10:]):
                            target[i, yg, xg, 5:10] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                        elif target[i, yg, xg, 4] == 0:
                            target[i, yg, xg, 5:10] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                            target[i, yg, xg, 10:] = clss
                '''
                # first come, first serve.
                if target[i, yg, xg, 4] == 0:
                    target[i, yg, xg, :5] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                    target[i, yg, xg, 10:] = clss
                elif target[i, yg, xg, 4] == 1.0 and clss.equal(target[i, yg, xg, 10:]) and target[i, yg, xg, 9] == 0:
                    target[i, yg, xg, 5:10] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                '''
        return target


class SimpleYOLOv1Loss(nn.Module):

    def __init__(self):
        super(SimpleYOLOv1Loss, self).__init__()

    def forward(self, pred, bboxes):
        """Compute loss with YOLOv1 interdoction
        Args:
            pred: (batch, grid, grid, boxes+classes)
            bboxes: List of array [b x (n, 5)]
        Return:
            loss: tensor float
        """
        device = pred.device
        target = self._make_target(device, bboxes)
        # obj_mask = target[:, :, :, [4, 9]]  # (?, ?, ?, 2)
        obj_mask = target[:, :, :, 4]  # (n, g, g)
        noobj_mask = (1.0 - obj_mask) * config.lamda_noobj
        # loss_contain = (target[:, :, :, [4, 9]] - pred[:, :, :, [4, 9]]) ** 2
        loss_contain = (target[:, :, :, 4] - pred[:, :, :, 4]) ** 2  # (n, g, g)
        loss_contain = (obj_mask + noobj_mask) * loss_contain
        # loss_contain = loss_contain.sum((1, 2, 3))
        loss_contain = loss_contain.sum((1, 2))  # (n,)
        # loss_coord = (
        #         (target[:, :, :, [0, 5]] - pred[:, :, :, [0, 5]]) ** 2 +
        #         (target[:, :, :, [1, 6]] - pred[:, :, :, [1, 6]]) ** 2 +
        #         (target[:, :, :, [2, 7]].sqrt() - pred[:, :, :, [2, 7]].sqrt()) ** 2 +
        #         (target[:, :, :, [3, 8]].sqrt() - pred[:, :, :, [3, 8]].sqrt()) ** 2
        # )
        loss_coord = (
                (target[:, :, :, 0] - pred[:, :, :, 0]) ** 2 +
                (target[:, :, :, 1] - pred[:, :, :, 1]) ** 2 +
                (target[:, :, :, 2].sqrt() - pred[:, :, :, 2].sqrt()) ** 2 +
                (target[:, :, :, 3].sqrt() - pred[:, :, :, 3].sqrt()) ** 2
        )  # (n, g, g)
        loss_coord = obj_mask * loss_coord * config.lamda_coor  # (n, g, g)
        # loss_coord = loss_coord.sum((1, 2, 3))
        loss_coord = loss_coord.sum((1, 2))  # (n,)
        # loss_class = (target[:, :, :, 10:] - pred[:, :, :, 10:]) ** 2
        loss_class = (target[:, :, :, 5:] - pred[:, :, :, 5:]) ** 2  # (n, g, g, 20)
        # loss_class = obj_mask.max(3, keepdim=True).values * loss_class
        # loss_class = loss_class.sum((1, 2, 3))
        loss_class = obj_mask.unsqueeze(3) * loss_class  # (n, g, g, 20)
        loss_class = loss_class.sum((1, 2, 3))
        return (loss_contain + loss_coord + loss_class).mean()

    def _make_target(self, device, bboxes_list):
        """make bbox to feature like
        Args:
            device: cpu or cuda index
            bboxes_list: List of array [b x (n, 5)]
        Return:
            target: (batch, grid, grid, boxes+classes)
        """
        # assert config.batch_size == len(bboxes_list)
        if config.batch_size != len(bboxes_list):
            batch_size = len(bboxes_list)
        else:
            batch_size = config.batch_size
        target = torch.zeros(
            (batch_size, config.grids, config.grids, 5*config.boxes_per_grid + config.classes),
            dtype=torch.float32,
            device=device
        )
        image_size = torch.FloatTensor(
            [config.image_size[0], config.image_size[1], config.image_size[0], config.image_size[1]]
        ).to(device)
        for i, bboxes in enumerate(bboxes_list):
            # bboxes: (?, 5)
            box = bboxes[:, :4] / image_size  # (?, 4) float32
            label = bboxes[:, 4].long()  # (?,) int64
            x_center = (box[:, 0] + box[:, 2]) * 0.5  # all pic
            y_center = (box[:, 1] + box[:, 3]) * 0.5
            w_box = box[:, 2] - box[:, 0]
            h_box = box[:, 3] - box[:, 1]
            class_target = F.one_hot(label, config.classes).float()  # (n, 20)
            x_grid = (x_center * config.grids).long()  # which grid
            y_grid = (y_center * config.grids).long()
            x_offset = x_center * config.grids - x_grid
            y_offset = y_center * config.grids - y_grid
            for xg, yg, xo, yo, w, h, clss in zip(x_grid, y_grid, x_offset, y_offset, w_box, h_box, class_target):
                '''
                # big first, small second.
                area = w * h
                if area > config.obj_area_threshold:
                    if target[i, yg, xg, 4] == 0:
                        target[i, yg, xg, :5] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                        if target[i, yg, xg, 9] == 1.0 and not clss.equal(target[i, yg, xg, 10:]):
                            # remove small obj
                            target[i, yg, xg, 5:10] = torch.FloatTensor([0, 0, 0, 0, 0]).to(device)
                            target[i, yg, xg, 10:] = clss
                        elif target[i, yg, xg, 9] == 0:
                            target[i, yg, xg, 10:] = clss
                else:
                    if target[i, yg, xg, 9] == 0:
                        if target[i, yg, xg, 4] == 1.0 and clss.equal(target[i, yg, xg, 10:]):
                            target[i, yg, xg, 5:10] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                        elif target[i, yg, xg, 4] == 0:
                            target[i, yg, xg, 5:10] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                            target[i, yg, xg, 10:] = clss

                # first come, first serve.
                if target[i, yg, xg, 4] == 0:
                    target[i, yg, xg, :5] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                    target[i, yg, xg, 10:] = clss
                elif target[i, yg, xg, 4] == 1.0 and clss.equal(target[i, yg, xg, 10:]) and target[i, yg, xg, 9] == 0:
                    target[i, yg, xg, 5:10] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                '''
                if target[i, yg, xg, 4] == 0:
                    target[i, yg, xg, :5] = torch.FloatTensor([xo, yo, w, h, 1.0]).to(device)
                    target[i, yg, xg, 5:] = clss
        return target


def test():
    yololoss = YOLOv1Loss()
    box = [torch.tensor([[0, 0, 448, 448, 0], [220, 220, 228, 228, 0]])]
    pred = torch.zeros(1, 7, 7, 30)
    pred[0, 3, 3, :11] = torch.tensor([0.5, 0.5, 1, 1, 1, 0.5, 0.5, 0.2, 0.2, 0.9, 1])
    loss = yololoss(pred, box)
    print(loss)


if __name__ == '__main__':
    test()
