# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:22:10 2021

@author:
"""
import os
import warnings
warnings.filterwarnings("ignore")

from yolo_inference.models.common import DetectMultiBackend
from yolo_inference.utils.augmentations import letterbox
from yolo_inference.utils.general import check_img_size, non_max_suppression, scale_boxes, scale_segments, xyxy2xywh
from yolo_inference.utils.segment.general import masks2segments, process_mask, process_mask_native
from yolo_inference.utils.torch_utils import select_device
import numpy as np
import torch

def masked(im0s, polygon, index):
    mask = np.zeros((im0s.shape[0], im0s.shape[1]), dtype =np.uint8)
    cv2.fillPoly(mask, pts =[np.int32(polygon)], color =index)
    # import matplotlib.pyplot as plt
    # plt.imshow(mask)
    # plt.show()
    return mask


class Segment(object):
    def __init__(self, cfg):
        weights, imgsz = cfg.weights, cfg.img_size
        imgsz *= 2 if len(imgsz) == 1 else 1  # expand

        self.device = select_device(cfg.device)
        self.augment, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, self.max_det =\
            cfg.augment, cfg.conf_thres, cfg.iou_thres, cfg.classes, cfg.agnostic_nms, cfg.max_det
        self.half = self.device.type != 'cpu' and os.path.splitext(weights)[-1] == '.pt'

        self.model = DetectMultiBackend(weights, device = self.device, dnn = False, data = None, fp16 = self.half)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.names = names
        self.imgsz = check_img_size(imgsz, s=stride)

        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup

    def __call__(self, im0s, confidence=0.5):
        img = letterbox(im0s, new_shape = self.imgsz)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        with torch.no_grad():
            # pred = self.model(im, augment = self.augment)
            pred, proto = self.model(im, augment = self.augment)[:2]
            torch.cuda.empty_cache()
        pred = non_max_suppression(pred, confidence, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det = self.max_det, nm=32)
        res = []
        for i, det in enumerate(pred):
            im0 = im0s
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample = True)
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                segments = [scale_segments(im.shape[2:], x, im0.shape, normalize = False)
                    for x in reversed(masks2segments(masks))]

                for index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    segment = segments[index]
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    bbox = [int(i.item()) for i in xyxy]
                    # quadrangle = [[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                    #               [bbox[2], bbox[3]], [bbox[0], bbox[3]]]

                    score = round(conf.item(), 2)
                    res.append({
                        "score": score,
                        "label_name": self.names[int(cls)],
                        # "box": xywh,
                        # "quadrangle": quadrangle,
                        "bbox": bbox,
                        'segment': segment.tolist()
                    })
        return res



if __name__ == '__main__':
    from common.params import args
    import cv2

    segment = Segment(args)

    file = r'test/youji/0001.jpg'
    img = cv2.imdecode(np.fromfile(file, dtype = np.uint8), cv2.IMREAD_COLOR)
    result = segment(img)
    print(result)

    # img_list = []
    # path = r'test/2'
    # for i in os.listdir(path):
    #     file = os.path.join(path, i)
    #     img = cv2.imdecode(np.fromfile(file, dtype = np.uint8), cv2.IMREAD_COLOR)
    #     img_list.append(img)
    #
    # result = detect.batch_inference(img_list)
    # print(result)
