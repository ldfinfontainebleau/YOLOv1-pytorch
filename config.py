# -*- coding: utf-8 -*-
"""
@author: LiuDanfeng
@since: 2021-09-30
"""

# data config
train_data = '/edgeai/shared/voc2007/train.ds'
test_data = '/edgeai/shared/voc2007/test.ds'
classes = 20
image_size = (448, 448)
classname = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# training config
pretrained = True  # for imagenet backbone
weights = 'checkpoints/best.pkl'
save_weight = 'checkpoints/best_nohup_2021_10_13.pkl'
gpu = '2'  # str 0,1.. or cpu
num_epochs = 100
batch_size = 64
freeze_backbone = True
max_lr = 2.5e-4
weight_decay = 0.3
optimizer = 'AdamW'
# evaulate config
iou_threshold = 0.5
evaulate_data = '/edgeai/shared/voc2007/test.ds'
save_result_folder = 'result_demo'
# net config
backbone = 'resnet18'  # resnet18， resnet50
grids = 7
boxes_per_grid = 2
head = 'detnet'
# decoder
obj_threshold = 0.1
nms_threshold = 0.5
# loss
lamda_coor = 5
lamda_noobj = 0.5
obj_area_threshold = 0.02
# ragular config
