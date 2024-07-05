# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import subprocess

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import json
import time
import logging
from PIL import Image
import ppocr.infer.utility as utility
import ppocr.infer.predict_rec as predict_rec
import ppocr.infer.predict_det as predict_det
import ppocr.infer.predict_cls as predict_cls
from ppocr.infer.utility import draw_ocr_box_txt, get_rotate_crop_image
from common.box_util import sort_bbox # todo
from common.ocr_utils import order_points
from math import atan, pi
from loguru import logger
from functools import cmp_to_key

class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)

        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)
        # dt_boxes = sorted_boxes(dt_boxes, min_y_overlap_ratio = 0.5)
        # if len(dt_boxes) != 0:  # todo
        #     dt_boxes = np.array(dt_boxes)
        #     dt_boxes = dt_boxes.reshape(dt_boxes.shape[0], -1)  # 修改 TODO
        #     _, dt_boxes, _, _ = sort_bbox(dt_boxes, min_y_overlap_ratio = 0.3)
        #     dt_boxes = np.array(dt_boxes, dtype = 'float32')
        #     dt_boxes = dt_boxes.reshape(dt_boxes.shape[0], -1, 2)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

# def coord_convert(bboxes):
#     # 4 points coord to 2 points coord for rectangle bbox
#     x_min, y_min, x_max, y_max = \
#         min(bboxes[0::2]), min(bboxes[1::2]), max(bboxes[0::2]), max(bboxes[1::2])
#     return [x_min, y_min, x_max, y_max]
#
# def _compare_box(box1, box2, min_y_overlap_ratio=0.5):
#     # 从上到下，从左到右
#     # box1, box2 to: [xmin, ymin, xmax, ymax]
#     box1 = coord_convert(box1.reshape(-1,1).squeeze().tolist())
#     box2 = coord_convert(box2.reshape(-1,1).squeeze().tolist())
#
#     def y_iou():
#         # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
#         # 判断是否有交集
#         if box1[3] <= box2[1] or box2[3] <= box1[1]:
#             return 0
#         # 计算交集的高度
#         y_min = max(box1[1], box2[1])
#         y_max = min(box1[3], box2[3])
#         return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))
#     yiou = y_iou()
#     if yiou > min_y_overlap_ratio:
#         return box1[0] - box2[0]
#     else:
#         return box1[1] - box2[1]
#
#
# def sorted_boxes(dt_boxes, min_y_overlap_ratio=0.5):
#     """
#     Sort resulting boxes in order from top to bottom, left to right
#     args:
#         dt_boxes(array): list of dict or tuple, box with shape [4, 2]
#     return:
#         sorted boxes(array): list of dict or tuple, box with shape [4, 2]
#     """
#     _boxes = sorted(dt_boxes, key=cmp_to_key(lambda x, y: _compare_box(x, y, min_y_overlap_ratio)))
#     return _boxes
