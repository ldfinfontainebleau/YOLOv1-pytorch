#!/usr/bin/env python3

"""
@author: LiuDanfeng
@since: 2021-09-16
"""

import torch
import numpy as np
# import sys 
# sys.path.append("..")
import config


class Evaluater(object):

    def __init__(self):
        self.classname = config.classname
        self.cls_gt = {}
        self.cls_pred = {}
        for cls_name in config.classname:
            self.cls_pred[cls_name] = {}
            self.cls_gt[cls_name] = {}

    def clear_pred(self):
        """Clear all pred.
        """
        self.cls_pred = {}
        for cls_name in config.classname:
            self.cls_pred[cls_name] = {}

    def preserve_pred(self, preds, imagenames):
        """save pred for every epoch test.
        save gt for onece.
        Args:
            pred: [b x (n, 6)] List of numpy.array.
            imagenames: [b x str] List of str.
        Returns:
            self.cls_pred[cls name][image name] = (m, 6) 
        """
        for pred, imagename in zip(preds, imagenames):
            for box in pred:
                if imagename not in self.cls_pred[config.classname[int(box[4])]].keys():
                    self.cls_pred[config.classname[int(box[4])]][imagename] = []
                self.cls_pred[config.classname[int(box[4])]][imagename].append(box)

    def preserve_gt(self, gts, imagenames):
        """save groundtruth for compute map.
        Args:
            gts: [b x (n, 5)] List of numpy.array
        """
        for gt, imagename in zip(gts, imagenames):
            for box in gt:
                if imagename not in self.cls_gt[config.classname[box[4].int()]].keys():
                    self.cls_gt[config.classname[box[4].int()]][imagename] = []
                self.cls_gt[config.classname[box[4].int()]][imagename].append(box.numpy() / config.image_size[0])

    def compute_map(self):
        """compute mAP
        Use numpy.array
        """
        mAP = list()
        # for each class compute its AP.
        for i, classname in enumerate(config.classname):
            ap = list()
            # print(self.cls_gt[classname].keys())
            # for each image compute its tp, pr, ap.
            for imagename in self.cls_gt[classname].keys():
                gt_boxes = self.cls_gt[classname][imagename]
                gt_boxes = np.array(gt_boxes)[:, :4]
                # TODO: pred is wrong, and gt does not have. gt == none
                if imagename in self.cls_pred[classname].keys():
                    pred_boxes = self.cls_pred[classname][imagename]
                    pred_boxes = np.array(pred_boxes)
                    pred_scores = pred_boxes[:, 5]
                    pred_boxes = pred_boxes[:, :4]
                else:
                    pred_boxes = np.empty(shape=[0, 4])
                    pred_scores = np.empty(shape=[0, 1])
                gt_num, tp, confid = self.calculate_tp(pred_boxes, pred_scores, gt_boxes)
                recall, precision = self.calculate_pr(gt_num, tp, confid)
                ap_pic = self.voc_ap(recall, precision)
                ap.append(ap_pic)
            mAP.append(np.mean(ap))
        self._print_result(mAP)
        return np.mean(mAP)

    def _print_result(self, mAP):
        """print result
        """
        for ap, class_ in zip(mAP, config.classname):
            print('---class {} ap {}---'.format(class_, ap))
        print('-map {}-'.format(np.mean(mAP)))

    @staticmethod
    def calculate_tp(pred_boxes, pred_scores, gt_boxes):
        """
            calculate tp/fp for all predicted bboxes for one class of one image.
            对于匹配到同一gt的不同bboxes，让score最高tp = 1，其它的tp = 0
        Args:
            pred_boxes: Numpy[N, 4], 某张图片中某类别的全部预测框的坐标 (x0, y0, x1, y1)
            pred_scores: Numpy[N, 1], 某张图片中某类别的全部预测框的score
            gt_boxes: Numpy[M, 4], 某张图片中某类别的全部gt的坐标 (x0, y0, x1, y1)
        Returns:
            gt_num: 某张图片中某类别的gt数量
            tp_list: 记录某张图片中某类别的预测框是否为tp的情况
            confidence_score: 记录某张图片中某类别的预测框的score值 (与tp_list相对应)
        """
        if np.size(gt_boxes) == 0:
            return 0, [], []

        # 若无对应的boxes，则 tp 为空
        if np.size(pred_boxes) == 0:
            return len(gt_boxes), [], []
        gt_num = len(gt_boxes)
        # 否则计算所有预测框与gt之间的iou
        ious = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=pred_boxes.dtype)
        for i in range(len(gt_boxes)):
            gb = gt_boxes[i]
            area_pb = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
            area_gb = (gb[2] - gb[0]) * (gb[3] - gb[1])
            xx1 = np.clip(pred_boxes[:, 0], gb[0], 1)  # [N-1,]
            yy1 = np.clip(pred_boxes[:, 1], gb[1], 1)
            xx2 = np.clip(pred_boxes[:, 2], 0, gb[2])
            yy2 = np.clip(pred_boxes[:, 3], 0, gb[3])
            inter = np.clip((xx2 - xx1), 0, 1) * np.clip((yy2 - yy1), 0, 1)  # [N-1,]
            ious[i] = inter / (area_pb + area_gb - inter)
        # 每个预测框的最大iou所对应的gt记为其匹配的gt
        max_ious = np.max(ious, axis=0)
        max_ious_idx = np.argmax(ious, axis=0)
        confidence_score = np.reshape(pred_scores, -1)
        tp_list = np.zeros_like(max_ious)
        for i in np.unique(max_ious_idx[max_ious > config.iou_threshold]):
            gt_mask = (max_ious > config.iou_threshold) * (max_ious_idx == i)
            idx = np.argmax(confidence_score * gt_mask)
            tp_list[idx] = 1
        return gt_num, tp_list.tolist(), confidence_score.tolist()

    @staticmethod
    def calculate_pr(gt_num, tp_list, confidence_score):
        """
        calculate all p-r pairs among different score_thresh for one class, using `tp_list` and `confidence_score`.

        Args:
            gt_num (Integer): 某张图片中某类别的gt数量
            tp_list (List): 记录某张图片中某类别的预测框是否为tp的情况
            confidence_score (List): 记录某张图片中某类别的预测框的score值 (与tp_list相对应)

        Returns:
            recall
            precision

        """
        if gt_num == 0:
            return [0], [0]
        if isinstance(tp_list, (tuple, list)):
            tp_list = np.array(tp_list)
        if isinstance(confidence_score, (tuple, list)):
            confidence_score = np.array(confidence_score)

        assert len(tp_list) == len(confidence_score), "len(tp_list) and len(confidence_score) should be same"

        if len(tp_list) == 0:
            return [0], [0]

        sort_mask = np.argsort(-confidence_score)
        tp_list = tp_list[sort_mask]
        recall = np.cumsum(tp_list) / gt_num
        precision = np.cumsum(tp_list) / (np.arange(len(tp_list)) + 1)

        return recall.tolist(), precision.tolist()

    @staticmethod
    def voc_ap(rec, prec, use_07_metric=False):
        """Compute VOC AP given precision and recall. If use_07_metric is true, uses
        the VOC 07 11-point method (default:False).
        """
        if isinstance(rec, (tuple, list)):
            rec = np.array(rec)
        if isinstance(prec, (tuple, list)):
            prec = np.array(prec)
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


def test():
    pred = [np.array([[0.35, 0.6, 0.6, 0.75, 1, 0.9], [0, 0, 0.25, 0.25, 0, 0.2]])]
    gt = [torch.tensor([[0.35, 0.6, 0.65, 0.85, 1], [0.05, 0.05, 0.125, 0.125, 0]])]
    ev = Evaluater()
    ev.preserve_pred(pred, ['1.jpg'])
    ev.preserve_gt(gt, ['1.jpg'])
    ap = ev.compute_map()
    print(ap)
    '''
    image_size = 400
    gt_num, tp, confid = ev.calculate_tp(
        bbox1/image_size,
        torch.tensor([[0.6], [0.31]]),
        bbox2/image_size,
        torch.tensor([0, 0])
    )
    recall, precision = ev.calculate_pr(gt_num, tp, confid)
    ap = ev.voc_ap(recall, precision)
    '''


if __name__ == '__main__':
    raise SystemExit(test())
