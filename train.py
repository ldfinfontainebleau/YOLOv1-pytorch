# -*- coding: utf-8 -*-
"""
@author: LiuDanfeng
@since: 2021-09-30
"""

import os
from re import S
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader

import config
from model.yolov1 import YOLOv1, Decoder, SimpleDecoder
from model.loss import YOLOv1Loss
# from utils.dataset import ObjectDetectionDataset
import utils.dataset as dataset
from utils.caculate_map import Evaluater
from utils.util import CosineWarmUpAnnealingLR


class Trainer(object):
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
        self._device = 'cpu' if config.gpu == 'cpu' else 'cuda'
        self.best_mAP = 0.
        self._load_data_set()
        self._create_model()
        self._create_optimizer()
        # self.decoder = SimpleDecoder()
        self.evaluator = Evaluater()
        self._load_weight()

        # self.loss = SimpleYOLOv1Loss().to(self._device)

    def _train_step(self, doc):
        self.model.train()
        pred = self.model(doc['image'].to(self._device))
        bboxes = [bbox.to(self._device) for bbox in doc['bboxes']]
        loss = self.loss(pred, bboxes)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._scheduler.step()
        return loss.detach().cpu(), self._scheduler.get_last_lr()[0]

    def _evaluate(self, epoch):
        self.model.eval()
        self.evaluator.clear_pred()
        loop = tqdm(self._test_loader, dynamic_ncols=True, leave=False)
        for doc in loop:
            pred = self.model(doc['image'].to(self._device))
            # TODO: add val loss?
            pred_boxes = self.decoder(pred)
            self.evaluator.preserve_pred(pred_boxes, doc['filename'])
            if epoch == 0:
                self.evaluator.preserve_gt(doc['bboxes'], doc['filename'])
        return self.evaluator.compute_map()

    def train(self):
        loss_g = 0.
        for epoch in range(config.num_epochs):
            loop = tqdm(self._train_loader, dynamic_ncols=True, leave=False)
            for doc in loop:
                loss, lr = self._train_step(doc)
                loss = float(loss.numpy())
                loss_g = 0.9 * loss_g + 0.1 * loss
                loop.set_description(f'[{epoch + 1}/{config.num_epochs}] L={loss_g:.06f} lr={lr:.01e}', False)
            mAP = self._evaluate(epoch)
            if mAP > self.best_mAP:
                self.best_mAP = mAP
                self._save_weight(config.save_weight, mAP)

    def _load_weight(self):
        """
        """
        if not config.pretrained:
            chkpt = torch.load(config.weights)
            self.model.load_state_dict(chkpt["model"])
            print('load weights from ', config.weights)

    def _create_model(self):
        self.model = YOLOv1().to(self._device)
        self.decoder = Decoder()

        # "requires_grad" of of the backbone parameters are set to False
        self._params_1 = [
            *self.model.head.parameters(),
        ]
        self._params_2 = [
            *self.model.backbone.parameters(),
        ]
        self.loss = YOLOv1Loss().to(self._device)

    def _create_optimizer(self):
        if config.freeze_backbone:
            param_groups = [
                {'params': self._params_1, 'lr': config.max_lr},
            ]
        else:
            param_groups = [
                {'params': self._params_1, 'lr': config.max_lr},
                {'params': self._params_2, 'lr': config.max_lr * 1e-2}
            ]
        if config.optimizer == 'MomentumSGD':
            self._optimizer = optim.SGD(
                param_groups,
                lr=config.max_lr,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        else:
            optimizer_class = getattr(optim, config.optimizer)
            self._optimizer = optimizer_class(
                param_groups,
                lr=config.max_lr,
                weight_decay=config.weight_decay,
            )
        self._scheduler = CosineWarmUpAnnealingLR(
            self._optimizer,
            config.num_epochs * len(self._train_loader)
        )

    def _save_weight(self, save_path, mAP) -> None:
        """Save trained model
        Args:
            save_path: model file saved in this path.
            epoch: int, epoch num.
        """
        chkpt = {
            "model": self.model.state_dict(),
            "mAP": mAP,
        }
        torch.save(chkpt, save_path)

    def _load_data_set(self):
        train_set_1 = dataset.ObjectDetectionDataset(config.train_data, train=True)
        train_set_2 = dataset.ObjectDetectionDataset('/edgeai/shared/voc2012/train.ds', train=True)
        train_set = train_set_1+train_set_2
        self._train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=train_set_1.collate_fn,
        )
        test_set = dataset.ObjectDetectionDataset(config.test_data, train=False)
        self._test_loader = DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=test_set.collate_fn,
        )


if __name__ == '__main__':
    raise SystemExit(Trainer().train())
