# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import re
from loading import rec, OCR, text_sys, e2e_algorithm



class Regular_match(object):
    def __init__(self, regulation_key):
        self.regulation_key = regulation_key

    def re_map(self, regulation, text):
        for i in regulation:
            res = re.search(i, text)
            if res:
                return res
        else:
            return None


    def find_code(self, dst_img):
        regu = self.regulation_key['code']
        text = rec(img_list = [dst_img])[0][0][0]
        rem = self.re_map(regu, text)
        if rem:
            result = rem.group().replace(' ','')
            return result.replace(' ','')
        else:
            return ""

    def find_num(self, dst_img):
        regu = self.regulation_key['num']
        text = rec(img_list = [dst_img])[0][0][0]
        rem = self.re_map(regu, text)
        if rem:
            result = rem.group()
            return result.replace(' ','')
        else:
            return ""

    def find_type(self, dst_img):
        ocr = OCR(text_sys, dst_img, cls = False, e2e_algorithm = e2e_algorithm)
        result = ocr(union = False, max_x_dist = 100, min_y_overlap_ratio = 0.5)
        type_ = [i['text'] for i in result]
        type_ = ''.join(type_)
        return type_.replace(' ','')

    def find_info(self, dst_img):
        ocr = OCR(text_sys, dst_img, cls = False, e2e_algorithm = e2e_algorithm)
        result = ocr(union = True, max_x_dist = 100, min_y_overlap_ratio = 0.5)
        text = [i['text'] for i in result]
        if len(text) >= 1:
            name2phone = text[0]
        else:
            name2phone = ''

        if len(text) >= 2:
            address = ''.join(text[1:])
        else:
            address = ''.join(text)
        return name2phone, address


    def __call__(self, dst_img, label, result):
        if label == 'num':
            num = self.find_num(dst_img)
            result['num'] = num
            return num
        elif label == 'code':
            code = self.find_code(dst_img)
            result[label] = code
            return code
        elif label == 'type':
            type_= self.find_type(dst_img)
            result[label] = type_
            return type_
        elif label == 'info':
            name2phone, address = self.find_info(dst_img)
            result['name2phone'] = name2phone
            result['address'] = address
            return name2phone+'\n'+address
