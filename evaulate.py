# -*- coding: utf-8 -*-
"""
@author: LiuDanfeng
@since: 2021-10-11
"""
import os
import cv2 as cv
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from model.yolov1 import YOLOv1, Decoder
import utils.dataset as dataset
from utils.dataset import decode_image
from utils.caculate_map import Evaluater


class Evaluate(object):

    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
        self._device = 'cpu' if config.gpu == 'cpu' else 'cuda'
        self.model = YOLOv1().to(self._device)
        self.decoder = Decoder()
        self.eva = Evaluater()
        self._load_weight()
        self._load_ds(config.evaulate_data)

    def _load_ds(self, ds_path: str):
        test_set = dataset.ObjectDetectionDataset(ds_path, train=False)
        self._test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=test_set.collate_fn,
        )

    def evaluate(self):
        self.model.eval()
        self.eva.clear_pred()
        loop = tqdm(self._test_loader, dynamic_ncols=True, leave=False)
        for doc in loop:
            pred = self.model(doc['image'].to(self._device))
            pred_boxes = self.decoder(pred)
            self.eva.preserve_pred(pred_boxes, doc['filename'])
            self.eva.preserve_gt(doc['bboxes'], doc['filename'])
            self._plot_result(pred_boxes, doc)
        return self.eva.compute_map()

    def _load_weight(self):
        chkpt = torch.load(config.weights)
        self.model.load_state_dict(chkpt["model"])

    @staticmethod
    def _plot_result(bbox, doc):
        image = doc['image'].numpy()[0]
        image = decode_image(image)
        h, w = image.shape[:2]
        image = np.ascontiguousarray(image, dtype=np.uint8)
        for i in range(len(bbox[0])):
            image = cv.rectangle(
                image,
                (int(bbox[0][i, 0] * w), int(bbox[0][i, 1] * h)),
                (int(bbox[0][i, 2] * w), int(bbox[0][i, 3] * h)),
                (30, 105, 210),
                thickness=1
            )
            image = cv.putText(
                image,
                config.classname[int(bbox[0][i, 4])],
                (int(bbox[0][i, 0] * w), int(bbox[0][i, 1] * h)),
                cv.FONT_HERSHEY_COMPLEX_SMALL,
                1.0,
                (0, 255, 0),
                thickness=1
            )
        cv.imwrite(config.save_result_folder + '/' + doc['filename'][0], image)


if __name__ == '__main__':
    raise SystemExit(Evaluate().evaluate())
