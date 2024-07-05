# -*- coding: utf-8 -*-
'''
Date:
Author:
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import platform
import torch


class Base_Config(object):
    def __init__(self, debug=True):
        self.use_gpu = True if torch.cuda.is_available() else False
        if self.use_gpu == True:
            self.device = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            self.device = 'cpu'

        self.ir_optim = True
        self.min_subgraph_size = 15
        self.precision = "fp32"
        self.gpu_mem = 500

        self.use_onnx = False #if platform.system().lower() == 'windows' or self.use_gpu == False else False

        # params for text detector
        self.remove_slip = False
        self.slip_angle = 5  # 度数：°

        self.det_algorithm = "DB" # todo
        if self.use_onnx:
            self.det_model_dir = "inference/ocr/det/det_v4.onnx"
        else:
            self.det_model_dir = "inference/ocr/det"

        self.det_limit_side_len = 1696 # 960
        self.det_limit_type = 'max'
        # limit_type='max', det_limit_side_len=960表示网络输入图像的最长边不能超过960，
        # 如果超过这个值，会对图像做等宽比的resize操作，确保最长边为det_limit_side_len

        # DB parmas
        self.det_db_thresh = 0.3  # DB模型输出预测图的二值化阈值
        self.det_db_box_thresh = 0.4  # DB模型输出框的阈值，低于此值的预测框会被丢弃
        self.det_db_unclip_ratio = 1.5  # 检测后处理时控制文本框大小,DB模型输出框扩大的比例
        self.use_dilation = False
        self.det_db_score_mode = "fast"

        # params for text recognizer
        self.rec_algorithm = "SVTR_LCNet" # todo
        self.rec_image_shape = "3, 48, 320"
        if self.use_onnx:
            self.rec_model_dir = "inference/ocr/rec/common/rec_v4.onnx"
        else:
            self.rec_model_dir = "inference/ocr/rec/common"

        self.rec_char_type = 'ch'
        self.rec_batch_num = 30  # 进行识别时，同时前向的图片数
        self.max_text_length = 25  # 默认训练时的文本可识别的最大长度

        self.rec_char_dict_path = "ppocr/utils/ppocr_keys_v1.txt" # todo
        self.use_space_char = True
        self.cand_alphabet = None # todo
        self.vis_font_path = "./ppocr/fonts/simfang.ttf"
        self.drop_score = 0.3
        if debug:
            self.is_visualize = True
            self.is_visualize_char = True
        else:
            self.is_visualize = False
            self.is_visualize_char = False

        self.save_crop_res = False
        self.crop_res_save_dir = r'./test/output'
        self.save_path = './test'

        # params for text classifier
        self.use_angle_cls = False
        self.cls_model_dir = "inference/ocr/cls/cls.onnx"
        self.cls_image_shape = "3, 48, 192"
        self.label_list = ['0', '180']
        self.cls_batch_num = 30
        self.cls_thresh = 0.9

        self.enable_mkldnn = False
        self.cpu_threads = 10
        self.use_pdserving = False
        self.warmup = False
        self.use_tensorrt = False

        self.use_mp = True
        self.total_process_num = 4
        self.process_id = 0
        self.ir_optim = True
        self.benchmark = False
        self.show_log = True

    def __getattr__(self, item):
        return item

class NumCode_Config(Base_Config):
    def __init__(self, debug=True):
        super(NumCode_Config, self).__init__(debug)
        # self.rec_char_dict_path = "./ppocr/utils/kuaidi_dict.txt"
        self.rec_char_dict_path = "./ppocr/utils/db_dict.txt"
        if self.use_onnx:
            self.rec_model_dir = "inference/ocr/rec/db/rec_v4_db.onnx"
        else:
            self.rec_model_dir = "inference/ocr/rec/db"

class Seg_Config(Base_Config):
    def __init__(self, debug=True):
        super(Seg_Config, self).__init__(debug)
        self.img_size = [640]
        self.iou_thres = 0.5
        self.classes = [1,3]
        self.augment = False
        self.agnostic_nms = False
        self.max_det = 1000
        self.weights = "inference/seg/best.pt"



debug = False
args = Base_Config(debug = debug)
args_numcode = NumCode_Config(debug=debug)
args_seg = Seg_Config(debug = debug)


