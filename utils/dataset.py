#!/usr/bin/env python3
"""
@author: LiuDanfeng
@since: 2021-09-15
"""
# import sys
# sys.path.append("..")
import config
import cv2 as cv
import imgaug.augmenters as iaa
import numpy as np
import torch
from docset import DocSet
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torch.utils.data import Dataset
from tqdm import tqdm

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
IGNORE_CLASS = 255


def encode_image(image: np.ndarray) -> np.ndarray:
    """Convert an image to float32 and CHW format.

    :param image: np.ndarray, dtype=uint8, shape=(h, w, c)
    :return: np.ndarray, dtype=float32, shape=(c, h, w)
    """
    image = image.astype(np.float32)
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    return image


def decode_image(tensor: np.ndarray) -> np.ndarray:
    """Convert float tensor back to an image.

    :param tensor: np.ndarray, dtype=float32, shape=(c, h, w)
    :return: np.ndarray, dtype=uint8, shape=(h, w, c)
    """
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = tensor * STD + MEAN
    tensor = np.clip(tensor, 0, 255)
    return tensor.astype(np.uint8)


def encode_bbox(bbox: np.ndarray) -> np.ndarray:
    return bbox.astype(np.int64)


def decode_label(tensor: np.ndarray) -> np.ndarray:
    return tensor.astype(np.uint8)


class ImageTransform:
    """Image Argument with imgaug.
    """

    def __init__(self, transform):
        super(ImageTransform, self).__init__()
        self._transform = transform

    def __call__(self, doc):
        image = doc['image']
        if isinstance(image, bytes):
            image = cv.imdecode(np.frombuffer(doc['image'], np.byte), cv.IMREAD_COLOR)
        assert isinstance(image, np.ndarray)
        bbs = BoundingBoxesOnImage([
            BoundingBox(
                x1=_object['box'][0],
                y1=_object['box'][1],
                x2=_object['box'][2],
                y2=_object['box'][3]
            ) for _object in doc['objects']],
            shape=image.shape
        )
        image, bbs = self._transform(image=image, bounding_boxes=bbs)
        bbox = [
            [
                bbs.bounding_boxes[i].x1,
                bbs.bounding_boxes[i].y1,
                bbs.bounding_boxes[i].x2,
                bbs.bounding_boxes[i].y2,
                doc['objects'][i]['label']
            ] for i in range(len(bbs))
        ]
        bbox = np.array(bbox)
        return image, bbox


class TestTransform(ImageTransform):

    def __init__(self, height, width):
        super(TestTransform, self).__init__(iaa.Sequential([
            iaa.Resize({'height': height, 'width': width})
        ]))


class TrainTransform(ImageTransform):
    def __init__(self, height, width):
        super(TrainTransform, self).__init__(iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            # iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
            # iaa.Affine(
            #     translate_px={"x": 40, "y": 60},
            #     scale=(0.5, 0.7),
            #     rotate=(-10, 10)
            # ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
            # iaa.GaussianBlur(sigma=(0.0, 3.0)),
            # iaa.Resize({'longer-side': config.image_size[0], 'shorter-side': 'keep-aspect-ratio'}),
            # iaa.PadToFixedSize(height=config.image_size[0], width=config.image_size[1])
            iaa.Resize({'height': height, 'width': width})
        ]))


class ObjectDetectionDataset(Dataset):

    def __init__(self, ds_path: str, train: bool):
        super(ObjectDetectionDataset, self).__init__()
        if train:
            self._transform = TrainTransform(config.image_size[0], config.image_size[1])
        else:
            self._transform = TestTransform(config.image_size[0], config.image_size[1])
        self.doc = DocSet(ds_path, 'r')

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, i):
        doc = self.doc[i]
        image, bbox = self._transform(doc)
        doc = {
            'filename': doc['filename'],
            'image': encode_image(image),
            'bboxes': bbox
        }
        return doc

    @staticmethod
    def collate_fn(doc_list):
        filename_list = []
        image_list = []
        bboxes_list = []
        for doc in doc_list:
            filename_list.append(doc['filename'])
            image_list.append(doc['image'])
            bboxes_list.append(doc['bboxes'])
        doc = {
            'filename': filename_list,
            'image': torch.from_numpy(np.stack(image_list)),
            'bboxes': [torch.from_numpy(bbox) for bbox in bboxes_list]
        }
        return doc


def test():
    from torch.utils.data import DataLoader
    ds = ObjectDetectionDataset('/edgeai/shared/voc2007/train.ds')
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    # print(len(loader))
    for doc in tqdm(loader):
        image = doc['image'].numpy()[0]
        image = decode_image(image)
        image = np.ascontiguousarray(image, dtype=np.uint8)
        bbox = doc['bboxes']
        for i in range(len(bbox[0])):
            cv.rectangle(
                image,
                (int(bbox[0][i, 0]), int(bbox[0][i, 1])),
                (int(bbox[0][i, 2]), int(bbox[0][i, 3])),
                (30, 105, 210),
                thickness=1
            )
            cv.putText(image, config.classname[int(bbox[0][i, 4])], (int(bbox[0][i, 0]), int(bbox[0][i, 1])),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0, 255, 0), thickness=1)
        cv.imwrite('show/'+doc['filename'][0], image)
    return 0


if __name__ == '__main__':
    raise SystemExit(test())
