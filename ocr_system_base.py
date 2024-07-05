# -*- coding: utf-8 -*-

'''
Date: 2020.10.21
Author:
'''
import os

from common.params import args

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cpu = args.use_gpu == False
if cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
from PIL import Image
from loguru import logger as log
from ppocr.infer.predict_system import TextSystem
from ppocr.infer.predict_e2e import TextE2E
from common.ocr_utils import string_to_arrimg, order_points, fourxy2twoxy
from common.exceptions import ParsingError
from ppocr.infer.utility import draw_ocr_box_txt, draw_e2e_res
import copy
from common.box_util import stitch_boxes_into_lines_v2 as stitch_boxes_into_lines

def load_model(args, e2e_algorithm=False):
    log.info("Loading model...")
    if args.use_gpu:
        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            log.info("use gpu: %s"%args.use_gpu)
            log.info("CUDA_VISIBLE_DEVICES: %s"%_places)
            args.gpu_mem = 500
        except:
            raise RuntimeError(
                "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
            )
    else:
        log.info("use gpu: %s"%args.use_gpu)
    if e2e_algorithm:
        text_sys = TextE2E(args)
    else:
        text_sys = TextSystem(args)
    # log.info(args.__dict__)
    # warm up 10 times
    # if args.warmup:
    #     img_warm = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
    #     for i in range(10):
    #         _ = text_sys(img_warm)
    return e2e_algorithm, text_sys


class OCR():
    def __init__(self, text_sys, img_str, cls = False, e2e_algorithm=False):
        if args.warmup:
            img_warm = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                _ = text_sys(img_warm)

        if isinstance(img_str,str):
            img = string_to_arrimg(img_str)
            if img is None:
                raise ParsingError('Failed to transform base64 to image.', 4)
        else:
            img = img_str

        self.img = img
        self.e2e_algorithm = e2e_algorithm
        if self.e2e_algorithm:
            self.dt_boxes, self.rec_res = text_sys(img)
        else:
            self.dt_boxes, self.rec_res = text_sys(img, cls)

    def __call__(self, union=False, max_x_dist = 50, min_y_overlap_ratio = 0.5, save_draw_img=None):
        img = self.img
        img_origin = copy.deepcopy(img)

        if self.e2e_algorithm:
            dt_boxes, rec_res = self.dt_boxes, self.rec_res
            dt_num = len(dt_boxes)
            result = []
            for dno in range(dt_num):
                result.append({"text": rec_res[dno], "box": (dt_boxes[dno]).tolist()})

            if args.is_visualize:
                src_im = draw_e2e_res(dt_boxes, rec_res, img)
                cv2.imwrite('./test/draw_img.jpg', src_im)
                log.info("The visualized image saved in ./test/draw_img.jpg")
        else:
            dt_boxes, rec_res = self.dt_boxes, self.rec_res
            dt_num = len(dt_boxes)
            result = []
            for dno in range(dt_num):
                text, score = rec_res[dno]

                quadrangle = dt_boxes[dno]
                temp_result = {"text": text,
                               "quadrangle": quadrangle.tolist(),
                               "box": quadrangle.reshape(1, -1).squeeze().tolist(),
                               "bbox": fourxy2twoxy(quadrangle),
                               "score": float(score)}

                result.append(temp_result)

            if union:
                result = stitch_boxes_into_lines(result, max_x_dist = max_x_dist,
                                                       min_y_overlap_ratio = min_y_overlap_ratio)

            if args.is_visualize:
                image = Image.fromarray(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB))
                if union:
                    boxes = [np.array(i['quadrangle']) for i in result]
                    txts = [i['text'] for i in result]
                    scores = [i['score'] for i in result]
                else:
                    boxes = dt_boxes
                    txts = [rec_res[i][0] for i in range(len(rec_res))]
                    scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=args.drop_score,
                    font_path=args.vis_font_path)
                if save_draw_img is None:
                    cv2.imwrite('./test/draw_img.jpg', draw_img[:, :, ::-1])
                else:
                    cv2.imwrite(save_draw_img, draw_img[:, :, ::-1])
                log.info("The visualized image saved in ./test/draw_img.jpg")

        return result


if __name__ == "__main__":
    import cv2, json
    from common.ocr_utils import imagefile_to_string
    from common.params import args, args_numcode

    e2e_algorithm, text_sys = load_model(args, e2e_algorithm = False)
    # log.info(args.__dict__)

    filename = r'test/1/01706246470841.jpg'
    filename = r'test/crop_image.jpg'
    img = imagefile_to_string(filename)
    # print(type(img))

    # img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    # print(type(img))
    ocr = OCR(text_sys, img, cls = False, e2e_algorithm = e2e_algorithm)
    result = ocr(union = False, max_x_dist = 100, min_y_overlap_ratio = 0.5)

    # with open(filename.rsplit('.', 1)[0] + '.json', 'w', encoding='utf8') as f:
    #     f.write(json.dumps(result, ensure_ascii=False, sort_keys=False, separators=(",", ":")))
    print(result)