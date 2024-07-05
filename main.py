
import os

import pandas as pd

from common.match_rec import match_by_iou, match_by_center
from common.params import args
from loguru import logger as log
from common.ocr_utils import fourxy2twoxy, cal_angle, text_area, string_to_arrimg
from common.angle_utils import get_img_rot_broa
from common.contour_detection import contour_detection, get_rotate_crop_image, minAreaRect
from common.regular_matching import Regular_match
from common.box_util import uncliped_bbox, convert_coord
from common.json_parse import jsonfile_to_dict
from common.timeit import time_it
import numpy as np
import cv2, random, copy
from common.plots import plot_one_box, Annotator
from loading import OCR, segment, text_detection, text_sys, e2e_algorithm
from common.exceptions import ParsingError



@time_it
def main(img_str, confidence=0.3, auto_rotate_whole_image=True):
    if isinstance(img_str, str):
        img = string_to_arrimg(img_str, log_flag = True)
        if img is None:
            raise ParsingError('Failed to transform base64 to image.', 2)
    else:
        img = img_str

    if auto_rotate_whole_image:
        dt_boxes_origin, _ = text_detection(img)
        angle = cal_angle(np.array(dt_boxes_origin), std_max = 10)
        log.info(f"tilt angle：{angle}")
        dt_boxes = [fourxy2twoxy(i) for i in dt_boxes_origin]
        img = text_area(img, dt_boxes)
        if angle != 0:
            img = get_img_rot_broa(img, -angle)
        if args.is_visualize:
            cv2.imwrite(r'test/adjust.jpg', img)

    result = {
        # "num": "",
        # "type": "",
        "name": "",
        "address": "",
        "phone": "",
        "code": "",
    }

    json_path = r'config/config.json'
    regulation_key = jsonfile_to_dict(json_path = json_path)
    rm = Regular_match(regulation_key)

    result_seg = segment(img, confidence = confidence)
    result_seg = sorted(result_seg, key = lambda x: x['score'])
    rec_keyword = {i['label_name']: i['segment'] for i in result_seg}

    ocr = OCR(text_sys, img, cls = False, e2e_algorithm = e2e_algorithm)
    ocr_result = ocr(union = False, max_x_dist = 100, min_y_overlap_ratio = 0.5)

    result_match = match_by_iou(img, ocr_result, rec_keyword, by_y = False, iou_threshold = 0.5)

    # temp = copy.deepcopy(img)
    annotator = Annotator(np.ascontiguousarray(img), line_width = 3, font_size=20)
    colors = {i: [random.randint(0, 255) for j in range(3)] for i in segment.names.values()}
    for r_s in result_seg:
        label = r_s['label_name']
        bbox = r_s['bbox']
        quadrangle = convert_coord(bbox)
        quadrangle = uncliped_bbox(quadrangle, unclip_ratio = 0.3, img_height = img.shape[0], img_width = img.shape[1])
        bbox = fourxy2twoxy(quadrangle)
        bbox = [int(i) for i in bbox]
        seg = r_s['segment']
        corner_points = minAreaRect(img, seg)
        if corner_points is None:
            dst_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        else:
            # corner_points = uncliped_bbox(corner_points, unclip_ratio = 0.5, img_height = img.shape[0],
            #                               img_width = img.shape[1])
            dst_img = get_rotate_crop_image(img, corner_points)
        if label in ['code']:
            res =  rm(dst_img, label, result)
        elif label == 'info':
            name = result_match.get('name')
            phone = result_match.get('phone')
            address = result_match.get('address')
            result['name'] = name if name else ""
            result['phone'] = phone if phone else ""
            result['address'] = address if address else ""
            res = str(name) + '\n' + str(phone) + '\n' + str(address)
        else:
            res = result_match.get(label)
            res = res.replace(' ', '') if res else ""
            result[label] = res if res else ""
        if args.is_visualize:
            cv2.imwrite(r'test/crop_image.jpg', dst_img)

            color = colors[label]
            # plot_one_box(corner_points, temp, color, label = res, line_thickness = 3)
            annotator.box_label(corner_points, label = f'{label}：'+res, color = tuple(color))
            draw_img = annotator.result()

    if args.is_visualize:
        cv2.imwrite(r'test/draw_img.jpg', draw_img)

    mapping = {'name': '姓名', 'address': '地址', 'phone': '电话', 'code': '单号'}
    result = {mapping[i]: result[i] for i in result}
    return result

def batch_main(img_path, excel_path=r'output/output.xlsx'):
    save_path = os.path.split(excel_path)[0]
    if len(save_path) > 0:
        os.makedirs(save_path, exist_ok = True)

    n = 0
    res_list = []
    for maindir, subdir, file_name_list in list(os.walk(img_path)):
        for file in file_name_list:
            if file.rsplit('.')[-1].lower() in ['bmp', 'png', 'jpg', 'jpeg']:
                n += 1
                apath = os.path.join(maindir, file)
                log.info('处理第[%d]个文件[%s]' % (n, apath))
                img = cv2.imdecode(np.fromfile(apath, dtype=np.uint8), cv2.IMREAD_COLOR)
                try:
                    res = main(img)
                    res_list.append(res)
                except Exception as e:
                    log.warning(f'错误：{e}')
    df = pd.DataFrame(res_list, index = range(1, len(res_list)+1))
    df.to_excel(excel_path)



if __name__ == "__main__":
    # filename = r'test/1/微信图片_20240218165103.jpg'
    # img = cv2.imdecode(np.fromfile(filename, dtype = np.uint8), cv2.IMREAD_COLOR)
    # result = main(img, auto_rotate_whole_image=False)
    # print(result)

    img_path = r'test/image'
    batch_main(img_path)