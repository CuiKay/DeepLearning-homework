import re
from loguru import logger as log
import numpy as np
import cv2
from common.params import args
from shapely.geometry import Polygon
from common.box_util import stitch_boxes_into_lines_v2
from common.ocr_utils import cal_angle, recalrotateposition, get_img_rot_broa, fourxy2twoxy

def coord_convert(bboxes):
    # 4 points coord to 2 points coord for rectangle bbox
    x_min, y_min, x_max, y_max = \
        min(bboxes[0::2]), min(bboxes[1::2]), max(bboxes[0::2]), max(bboxes[1::2])
    return [x_min, y_min, x_max, y_max]

def convert_coord(xyxy):
    """
    Convert two points format to four points format.
    :param xyxy:
    :return:
    """
    new_bbox = np.zeros([4,2], dtype=np.float32)
    new_bbox[0,0], new_bbox[0,1] = xyxy[0], xyxy[1]
    new_bbox[1,0], new_bbox[1,1] = xyxy[2], xyxy[1]
    new_bbox[2,0], new_bbox[2,1] = xyxy[2], xyxy[3]
    new_bbox[3,0], new_bbox[3,1] = xyxy[0], xyxy[3]
    return new_bbox

def match_by_iou(ori_img, ocr_result, rec_keyword, by_y=False, iou_threshold=0.3):
    def y_iou(box1, box2):
        # 从上到下，从左到右
        # box1, box2 to: [xmin, ymin, xmax, ymax]
        # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
        # 判断是否有交集
        box1 = coord_convert(np.float16(box1).reshape(-1, 1).squeeze().tolist())
        box2 = coord_convert(np.float16(box2).reshape(-1, 1).squeeze().tolist())
        if box1[3] <= box2[1] or box2[3] <= box1[1] or box1[2] <= box2[0] or box2[2] <= box1[0]:
            return 0
        # 计算交集的高度
        y_min = max(box1[1], box2[1])
        y_max = min(box1[3], box2[3])
        return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))

    def cal_iou(bbox1, bbox2):
        bbox1_poly = Polygon(bbox1).convex_hull
        bbox2_poly = Polygon(bbox2).convex_hull
        # union_poly = np.concatenate((bbox1, bbox2))

        if not bbox1_poly.intersects(bbox2_poly):
            iou = 0
        else:
            inter_area = bbox1_poly.intersection(bbox2_poly).area
            # union_area = MultiPoint(union_poly).convex_hull.area
            smaller_area = min(bbox1_poly.area, bbox2_poly.area)
            if smaller_area == 0:
                iou = 0
            else:
                iou = float(inter_area) / smaller_area
        return iou

    temp_list  = []
    res = {}
    for i in range(len(rec_keyword)):
        res[list(rec_keyword.keys())[i]] = ""
    for rec_k, rec_v in rec_keyword.items():
        if len(rec_v) == 4:
            rec_v = convert_coord(rec_v).tolist()
        for dic in ocr_result:
            text = dic['text']
            quadrangle = dic['quadrangle']
            if by_y:
                iou = y_iou(rec_v, quadrangle)
            else:
                iou = cal_iou(rec_v, quadrangle)
            if iou >= iou_threshold:
                res[rec_k] += text
                if rec_k == 'info':
                    temp_list.append(dic)
    dt_boxes = np.float16([i['quadrangle'] for i in temp_list])
    tilt = cal_angle(dt_boxes)
    log.info(f"tilt angle：{tilt}")
    new_img = get_img_rot_broa(ori_img, -tilt)
    if args.is_visualize:
        cv2.imwrite(r'test/adjust.jpg', new_img)
    new_temp_list = []
    for i in temp_list:
        quad = i['quadrangle']
        quadrangle = np.array(
            [recalrotateposition(i[0], i[1], tilt, ori_img, new_img) for i in quad])
        dic = {"text": i['text'],
               'quadrangle': quadrangle.tolist(),
               'box': sum(quadrangle.tolist(), []),
               'bbox': fourxy2twoxy(quadrangle),
               'score': i['score']}
        new_temp_list.append(dic)

    temp_list = stitch_boxes_into_lines_v2(new_temp_list, max_x_dist = 200, min_y_overlap_ratio = 0.8)
    if 'info' in rec_keyword.keys():
        text_list = [i['text'] for i in temp_list]
        text = ' '.join(text_list)#.replace(' ', '')
        text_split = re.search('\d+[\u4e00-\u9fa5\s]', text)
        if text_split:
            text_split_position = text_split.end()
            name2phone = text[:text_split_position-1]
            address = text[text_split_position-1:]
        else:
            if len(text_list) >= 1:
                name2phone = text_list[0]
            else:
                name2phone = ''

            if len(text_list) >= 2:
                address = ''.join(text_list[1:])
            else:
                address = ''.join(text_list)
        res.pop('info')
    else:
        name2phone, address = "", ""
    if ' ' in name2phone:
        name = name2phone.split(' ')[0]
        phone = name2phone.split(' ')[1]
    else:
        re_ = re.search('[\d]+', name2phone)
        if re_:
            name = name2phone[:re_.start()]
            phone = name2phone[re_.start():]
            if '*' not in phone:
                name = name.strip('*') + '*'
                len_ = len(''.join(re.findall('\*', name2phone)))
                phone = (len_-1)*'*' + name2phone[re_.start():]
        else:
            name = name2phone
            phone = name2phone
    res.update({"name": name.replace('收', '').replace(' ', ''),
                "phone": phone.replace(' ', ''),
                'address': address.strip().replace(' ', '')}
               )
    return res

def make_template(childImg, final_rboxes):
    rectangle_dict = {}
    for box in range(len(final_rboxes)):
        rectangle_dict[box + 1] = np.int32(final_rboxes[box]).reshape(-1,2)

    template = np.zeros(childImg.shape[:2], dtype = 'uint16')
    for r in rectangle_dict:
        cv2.fillConvexPoly(template, rectangle_dict[r], r)
    return template, rectangle_dict

def match_by_center(ocr_result, rec_keyword, img):
    final_rboxes = list(rec_keyword.values())
    final_rboxes = [convert_coord(i) if len(i) == 4 else i for i in final_rboxes]
    template, rectangle_dict = make_template(img, final_rboxes)
    content_boxes_index = list(rectangle_dict.keys())
    res = {}
    for i in content_boxes_index:
        res[list(rec_keyword.keys())[i - 1]] = ""
    for idx, m in enumerate(ocr_result):
        point = [int(m["bbox"][0] + m["bbox"][2]) // 2, int(m["bbox"][1] + m["bbox"][3]) // 2]  # 中心点
        text = m["text"]
        label_ind = template[point[1]][point[0]]
        if label_ind in content_boxes_index:
            res[list(rec_keyword.keys())[label_ind - 1]] += text
    return res


def character_segmentation(ocr_result):
    char_result = []
    for dict in ocr_result:
        char_result.extend(dict['chars'])
    return char_result