# # Copyright (c) OpenMMLab. All rights reserved.
import copy
import re, cv2
import numpy as np
from common.match import is_inside

from common.ocr_utils import fourxy2twoxy
from common.match import convert_coord
from shapely.geometry import Polygon
import pyclipper

def coord_convert(bboxes):
    # 4 points coord to 2 points coord for rectangle bbox
    x_min, y_min, x_max, y_max = \
        min(bboxes[0::2]), min(bboxes[1::2]), max(bboxes[0::2]), max(bboxes[1::2])
    return [x_min, y_min, x_max, y_max]

def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    """Check if two boxes are on the same line by their y-axis coordinates.

    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.

    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    if a_y_min >= b_y_min and a_y_max <= b_y_max:  # todo
        return True

    if b_y_min >= a_y_min and b_y_max <= a_y_max:  # todo
        return True

    # Make sure that box a is always the box above another
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    # if b_y_min <= a_y_max and np.min(box_a[0::2]) != np.min(box_b[0::2]): # todo 新增and np.min(box_a[0::2]) != np.min(box_b[0::2])
    if b_y_min <= a_y_max and box_a[0] != box_b[0]:  # todo 新增and box_a[0] != box_b[0]
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or \
                overlap >= min_b_overlap
        else:
            return True
    return False


def stitch_boxes_into_lines(boxes, max_x_dist=10, min_y_overlap_ratio=0.8):
    """Stitch fragmented boxes of words into lines.

    Note: part of its logic is inspired by @Johndirr
    (https://github.com/faustomorales/keras-ocr/issues/22)

    Args:
        boxes (list): List of ocr results to be stitched
        max_x_dist (int): The maximum horizontal distance between the closest
                    edges of neighboring boxes in the same line
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                    allowed for any pairs of neighboring boxes in the same line

    Returns:
        merged_boxes(list[dict]): List of merged boxes and texts
    """

    if len(boxes) <= 1:
        return boxes

    merged_boxes = []

    # sort groups based on the x_min coordinate of boxes
    x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
    # store indexes of boxes which are already parts of other lines
    skip_idxs = set()

    i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            if j in skip_idxs:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                               x_sorted_boxes[j]['box'], min_y_overlap_ratio):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        lines = []
        line_idx = 0
        lines.append([line[0]])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k - 1]]
            dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])

        # Get merged boxes
        for box_group in lines:
            merged_box = {}
            merged_box['text'] = ' '.join([x_sorted_boxes[idx]['text'] for idx in box_group])

            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            origin_box = []
            score_lst = []
            for idx in box_group:
                score_lst.append(x_sorted_boxes[idx]['score'])
                origin_box.append(x_sorted_boxes[idx]['box'])
                x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
                x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
                y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
                y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)

            merged_box['quadrangle'] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            merged_box['box'] = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            merged_box['origin_box'] = origin_box

            charslist = []
            for idx in box_group:
                charslist.extend(x_sorted_boxes[idx].get('chars', []))
            if charslist != []:
                merged_box['chars'] = charslist
            merged_box['score'] = np.mean(score_lst)

            merged_boxes.append(merged_box)

    bboxes = [i['box'] for i in merged_boxes] # Todo
    bboxes = np.array(bboxes)
    bboxes = sorted_boxes(bboxes)

    index_dict = {f'{k}': i for i, k in enumerate(bboxes)}
    merged_boxes = sorted(merged_boxes, key = lambda x: index_dict[f"{x['box']}"])

    # bboxes_sorted = [None] * len(bboxes)
    # for bg_item in merged_boxes:
    #     idx = bboxes.index(bg_item['box'])
    #     bboxes_sorted[idx] = bg_item

    return merged_boxes


def stitch_boxes_into_lines_v2(boxes, max_x_dist=10, min_y_overlap_ratio=0.8, symbol=' ', concat=False):
    """Stitch fragmented boxes of words into lines.

    Note: part of its logic is inspired by @Johndirr
    (https://github.com/faustomorales/keras-ocr/issues/22)

    Args:
        boxes (list): List of ocr results to be stitched
        max_x_dist (int): The maximum horizontal distance between the closest
                    edges of neighboring boxes in the same line
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                    allowed for any pairs of neighboring boxes in the same line

    Returns:
        merged_boxes(list[dict]): List of merged boxes and texts
    """

    if len(boxes) <= 1:
        return boxes

    merged_boxes = []

    # sort groups based on the x_min coordinate of boxes
    if concat:
        x_sorted_boxes = sorted(boxes, key = lambda x: np.min(x['box'][1::2]))
    else:
        x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
    # store indexes of boxes which are already parts of other lines
    skip_idxs = set()

    # i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        bbox = x_sorted_boxes[i]['bbox']
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h / w * 1.0 > 2:  # todo, 竖直文本不合并
            x_sorted_boxes[i].update({'origin_box': [x_sorted_boxes[i]['box']]}) #  todo 解决缺失origi_box的bug 20221103
            merged_boxes.append(x_sorted_boxes[i])
            skip_idxs.add(i)
            continue
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            bbox = x_sorted_boxes[j]['bbox']
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h / w * 1.0 > 2:  # todo, 竖直文本不合并
                skip_idxs.add(j)
                continue
            if j in skip_idxs:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                               x_sorted_boxes[j]['box'], min_y_overlap_ratio):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        lines = []
        line_idx = 0
        lines.append([line[0]])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k - 1]]
            dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])

        # Get merged boxes
        for box_group in lines:
            merged_box = {}
            merged_box['text'] = symbol.join([x_sorted_boxes[idx]['text'] for idx in box_group])

            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            origin_box = []
            score_lst = []
            for idx in box_group:
                score_lst.append(x_sorted_boxes[idx]['score'])
                origin_box.append(x_sorted_boxes[idx]['box'])
                x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
                x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
                y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
                y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)

            merged_box['quadrangle'] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            merged_box['box'] = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            merged_box['bbox'] = [x_min, y_min, x_max, y_max]
            merged_box['origin_box'] = origin_box

            charslist = []
            for idx in box_group:
                charslist.extend(x_sorted_boxes[idx].get('chars', []))
            if charslist != []:
                merged_box['chars'] = charslist
            merged_box['score'] = np.mean(score_lst)

            merged_boxes.append(merged_box)

    bboxes = [i['box'] for i in merged_boxes] # Todo
    bboxes = np.array(bboxes)
    end2end_sorted_idx_list, end2end_sorted_bbox_list, sorted_groups, sorted_bbox_groups = sort_bbox(bboxes, min_y_overlap_ratio, concat = concat)

    # index_dict = {f'{k}': i for i, k in enumerate(end2end_sorted_bbox_list)}
    # new_merged_boxes = sorted(merged_boxes, key = lambda x: index_dict[f"{x['box']}"])

    new_merged_boxes = [None] * len(end2end_sorted_bbox_list)
    for bg_item in merged_boxes:
        idx = end2end_sorted_bbox_list.index(bg_item['box'])
        new_merged_boxes[idx] = bg_item

    return new_merged_boxes


def stitch_boxes_into_lines_invoiceservice(boxes, max_x_dist=10, min_y_overlap_ratio=0.8):
    if len(boxes) <= 1:
        boxes = [[b[0], np.array(coord_convert(b[2]), dtype = 'int32').reshape(2,2).tolist()] for b in boxes]
        return boxes

    merged_boxes = []

    # sort groups based on the x_min coordinate of boxes
    x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x[2][::2]))
    # store indexes of boxes which are already parts of other lines
    skip_idxs = set()

    # i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            if j in skip_idxs:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx][2],
                               x_sorted_boxes[j][2], min_y_overlap_ratio):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        lines = []
        line_idx = 0
        lines.append([line[0]])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k - 1]]
            dist = np.min(curr_box[2][::2]) - np.max(prev_box[2][::2])
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])

        # Get merged boxes
        for box_group in lines:
            merged_box = []
            merged_box.append(' '.join([x_sorted_boxes[idx][0] for idx in box_group]))

            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            for idx in box_group:
                x_max = max(np.max(x_sorted_boxes[idx][2][::2]), x_max)
                x_min = min(np.min(x_sorted_boxes[idx][2][::2]), x_min)
                y_max = max(np.max(x_sorted_boxes[idx][2][1::2]), y_max)
                y_min = min(np.min(x_sorted_boxes[idx][2][1::2]), y_min)

            merged_box.append([x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max])

            merged_boxes.append(merged_box)

    bboxes = [i[1] for i in merged_boxes] # Todo
    bboxes = np.array(bboxes)
    end2end_sorted_idx_list, end2end_sorted_bbox_list, sorted_groups, sorted_bbox_groups = sort_bbox(bboxes, min_y_overlap_ratio)

    # index_dict = {f'{k}': i for i, k in enumerate(end2end_sorted_bbox_list)}
    # new_merged_boxes = sorted(merged_boxes, key = lambda x: index_dict[f"{x['box']}"])

    new_merged_boxes = [None] * len(end2end_sorted_bbox_list)
    for bg_item in merged_boxes:
        idx = end2end_sorted_bbox_list.index(bg_item[1])
        new_merged_boxes[idx] = bg_item
    new_merged_boxes = [[b[0], np.array(coord_convert(b[1]), dtype = 'int32').reshape(2,2).tolist()] for b in new_merged_boxes]
    return new_merged_boxes

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[1], x[0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][1] - _boxes[i][1]) < 10 and \
                (_boxes[i + 1][0] < _boxes[i][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    _boxes = [i.tolist() for i in _boxes]
    return _boxes



def is_abs_lower_than_threshold(this_bbox, target_bbox, threshold=3):
    # only consider y axis, for grouping in row.
    delta = abs(this_bbox[1] - target_bbox[1])
    if delta < threshold:
        return True
    else:
        return False


def sort_line_bbox(g, bg, concat=False):
    """
    Sorted the bbox in the same line(group)
    compare coord 'x' value, where 'y' value is closed in the same group.
    :param g: index in the same group
    :param bg: bbox in the same group
    :return:
    """
    # todo 修改，避免sorted_xs中x值相同
    if concat:
        xs = [bg_item[1] for bg_item in bg]
    else:
        xs = [bg_item[0] for bg_item in bg]
    xs = [str(index) + '_' + str(i) for index, i in enumerate(xs)]
    sorted_xs = sorted(xs, key = lambda x: float(x.split('_')[1]))

    g_sorted = [None]*len(xs)
    bg_sorted = [None]*len(xs)
    for index, (g_item, bg_item) in enumerate(zip(g, bg)):
        # idx = xs_sorted.index(bg_item[0])
        if concat:
            idx = sorted_xs.index(str(index) + '_' + str(bg_item[1]))
        else:
            idx = sorted_xs.index(str(index) + '_' + str(bg_item[0]))
        bg_sorted[idx] = bg_item
        g_sorted[idx] = g_item

    return g_sorted, bg_sorted


def flatten(sorted_groups, sorted_bbox_groups):
    idxs = []
    bboxes = []
    for group, bbox_group in zip(sorted_groups, sorted_bbox_groups):
        for g, bg in zip(group, bbox_group):
            idxs.append(g)
            bboxes.append(bg.tolist())
    return idxs, bboxes


def sort_bbox(end2end_xyxy_bboxes, min_y_overlap_ratio = 0.2, concat=False):
    """
    This function will group the render end2end bboxes in row.
    :param end2end_xyxy_bboxes:
    :return:
    """
    groups = []
    bbox_groups = []
    for index, end2end_xywh_bbox in enumerate(end2end_xyxy_bboxes):
        this_bbox = end2end_xywh_bbox
        temp = coord_convert(this_bbox)
        w, h = temp[2] - temp[0], temp[3] - temp[1]
        if len(groups)==0 or h / w > 2.0: #  todo
            groups.append([index])
            bbox_groups.append([this_bbox])
        else:
            flag = False
            for g, bg in zip(groups, bbox_groups):
                temp = coord_convert(bg[0]) # todo
                w, h = temp[2] - temp[0], temp[3] - temp[1]
                if h / w > 2.0:
                    continue
                # this_bbox is belong to bg's row or not
                # if is_abs_lower_than_threshold(this_bbox, bg[0], threshold=threshold):
                if is_on_same_line(this_bbox, bg[0], min_y_overlap_ratio = min_y_overlap_ratio):
                    g.append(index)
                    bg.append(this_bbox)
                    flag = True
                    break
            if not flag:
                # this_bbox is not belong to bg's row, create a row.
                groups.append([index])
                bbox_groups.append([this_bbox])

    # sorted bboxes in a group
    tmp_groups, tmp_bbox_groups = [], []
    for g, bg in zip(groups, bbox_groups):
        g_sorted, bg_sorted = sort_line_bbox(g, bg, concat)
        tmp_groups.append(g_sorted)
        tmp_bbox_groups.append(bg_sorted)

    # sorted groups, sort by coord y's value.
    # sorted_groups = [None]*len(tmp_groups)
    # sorted_bbox_groups = [None]*len(tmp_bbox_groups)
    # ys = [bg[0][1] for bg in tmp_bbox_groups]
    # sorted_ys = sorted(ys)
    # for g, bg in zip(tmp_groups, tmp_bbox_groups):
    #     idx = sorted_ys.index(bg[0][1])
    #     sorted_groups[idx] = g
    #     sorted_bbox_groups[idx] = bg

    # todo 修改，避免sorted_ys中y值相同
    sorted_groups = [None]*len(tmp_groups)
    sorted_bbox_groups = [None]*len(tmp_bbox_groups)
    ys = [bg[0][1] for bg in tmp_bbox_groups]
    ys = [str(index) + '_' + str(i) for index, i in enumerate(ys)]
    sorted_ys = sorted(ys, key = lambda x: float(x.split('_')[1]))
    for index, (g, bg) in enumerate(zip(tmp_groups, tmp_bbox_groups)):
        # idx = sorted_ys.index(bg[0][1])
        idx = sorted_ys.index(str(index) + '_'+ str(bg[0][1]))
        sorted_groups[idx] = g
        sorted_bbox_groups[idx] = bg

    # flatten, get final result
    end2end_sorted_idx_list, end2end_sorted_bbox_list \
        = flatten(sorted_groups, sorted_bbox_groups)

    return end2end_sorted_idx_list, end2end_sorted_bbox_list, sorted_groups, sorted_bbox_groups


def fourxy2eightxy(dt_boxes):
    new_dt_boxes = dt_boxes.reshape(dt_boxes.shape[0], -1)
    return new_dt_boxes

def eightxy2fourxy(dt_boxes):
    new_dt_boxes = dt_boxes.reshape(dt_boxes.shape[0], -1, 2)
    return new_dt_boxes

def get_y_list(detail_dict_20):
    y_list = []
    for detail in detail_dict_20:
        text, rect_origin, rect = detail
        ymin, ymax = rect[0][1], rect[1][1]
        y_list.append([ymin, ymax])

    return y_list

def ave_y_list(y_list):
    if len(y_list) == 2:
        ave = int(np.mean([y_list[0][1], y_list[1][0]]))
        y_list = [[y_list[0][0], ave],
                  [ave, y_list[1][1]]]
    else:
        for idx, y in enumerate(y_list):
            if len(y_list) - 2 >= idx >= 1:
                ymin = (y_list[idx][0] + y_list[idx - 1][1]) // 2
                y_list[idx][0] = ymin
                y_list[idx - 1][1] = ymin
                ymax = (y_list[idx][1] + y_list[idx + 1][0]) // 2
                y_list[idx][1] = ymax
                y_list[idx + 1][0] = ymax
            # elif idx == 0:
            #     y_list[idx][1] = (y_list[idx][1] + y_list[idx+1][0])//2
            # elif idx == len(y_list)-1:
            #     y_list[idx][0] = (y_list[idx][0] + y_list[idx-1][1])//2
    return y_list

def partition(detail_dict_20, rectangle_dict_20_28, y_list=None):
    if y_list is None:
        y_list = get_y_list(detail_dict_20)
        y_list = ave_y_list(y_list)

    # rectangle_list = list(rectangle_dict.values())[20:28]
    rectangle_list = rectangle_dict_20_28
    lst = []
    for i in range(len(rectangle_list)):
        rect = []
        xmin, xmax = rectangle_list[i][0][0], rectangle_list[i][1][0]
        for j in range(len(y_list)):
            temp = [[xmin, y_list[j][0]], [xmax, y_list[j][1]]]
            rect.append(temp)
        lst.append(rect)
    return lst

def union_y(detail_dict_20, rectangle_dict_20):
    '''
    多行判断
    Args:
        detail_dict_20:
        rectangle_dict_20:

    Returns:

    '''
    xmax = rectangle_dict_20[1][0]
    flag_list = []
    for detail in detail_dict_20:
        text, rect = detail
        x_right = rect[1][0]
        if xmax - x_right <= 28: # todo 参数：明细文本框xmax与右侧框的距离
            flag_list.append(1)
        else:
            flag_list.append(0)

    if len(set(flag_list)) == 1:
        union = False
        return [[d[0], d[1], d[1]] for d in detail_dict_20]
    else:
        union = True
        flag_list = [str(i) for i in flag_list]
        flag_list = ''.join(flag_list)
        index_range = [i.span() for i in re.finditer('10', flag_list)]
        index_range = [(i[0], i[1]-1) for i in index_range]

        index_range_new = []
        for r in index_range:
            r_first = detail_dict_20[r[0]]
            r_second = detail_dict_20[r[1]]
            if (r_second[1][0][1] - r_first[1][1][1]) < 10: # 计算下一个的ymin与上一个的ymax的差值
                index_range_new.append(r)
        if len(index_range_new) == 0:
            return [[d[0], d[1], d[1]] for d in detail_dict_20]
        else:
            for r in index_range_new:
                detail_dict_20[r[0]][0] += '\n' + detail_dict_20[r[1]][0]
                detail_dict_20[r[0]][1] += detail_dict_20[r[1]][1]
                temp = np.array(detail_dict_20[r[0]][1], dtype = 'int32')
                temp = [[temp[:,0].min(),temp[:,1].min()],[temp[:,0].max(), temp[:,1].max()]]
                detail_dict_20[r[0]].append(temp)

            detail_dict_20_new = []
            pop_list = [i[1] for i in index_range_new]
            for idx, value in enumerate(detail_dict_20):
                if idx in pop_list:
                    continue
                detail_dict_20_new.append(value)

            max_len = max([len(i) for i in detail_dict_20_new])
            for i in detail_dict_20_new:
                if len(i) != max_len:
                    i.append(i[1])

            return detail_dict_20_new


def details_process(json_info, rectangle_dict, is_editpdf=False):
    json_info_temp = copy.deepcopy(json_info)
    rectangle_dict_20_28 = [v for k,v in rectangle_dict.items() if k in range(20,28)]
    rectangle_dict_details = np.array(rectangle_dict_20_28).reshape(-1,2)
    rect = [[rectangle_dict_details[:,0].min(), rectangle_dict_details[:,1].min()],
            [rectangle_dict_details[:,0].max(), rectangle_dict_details[:,1].max()]]

    details_info = []
    for i in json_info_temp:
        bbox = i['bbox']
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]  # todo
        if h * 1.0 / w > 2.0 and not is_editpdf:
            for char in i["chars"]:
                point = [int(char["bbox"][0] + char["bbox"][2]) // 2, int(char["bbox"][1] + char["bbox"][3]) // 2]
                flag = is_inside(point, rect)
                if flag:
                    char.update({'score': i['score']})
                    details_info.append(char)
        else:
            point = [(bbox[0]+bbox[2])/2, (bbox[3]+bbox[1])/2]
            flag = is_inside(point, rect)
            if flag:
                details_info.append(i)

    details_info_union = stitch_boxes_into_lines_v2(details_info, max_x_dist = 1000, min_y_overlap_ratio=0.5)

    rows = [i for i in details_info_union if len(i['text'].split(' ')) >= 3] # todo
    rows_bbox = [np.int32(i['bbox']).reshape(2,2).tolist() for i in rows]
    y_list = [[i[0][1],i[1][1]] for i in rows_bbox]

    # rectangle_dict_20_ymax = rectangle_dict[20][1][1]
    bboxes = [i['bbox'] for i in details_info_union]
    if len(bboxes) == 0:
        # rectangle_dict_20 = rectangle_dict[20]
        # y_list = [[rectangle_dict_20[0][1], rectangle_dict_20[1][1]]]
        end_index = list(rectangle_dict.keys())[-1]
        return 1, end_index, {20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [], 27: []}, rectangle_dict

    rectangle_dict_20_ymax = np.int32(bboxes)[:,-1].max()
    if len(y_list) == 2:
        max_value = int(np.max([y_list[0][1], y_list[1][0]]))
        y_list = [[y_list[0][0], max_value],
                  [max_value, rectangle_dict_20_ymax]]
    elif len(y_list) <= 1:
        rectangle_dict_20 = rectangle_dict[20]
        y_list = [[rectangle_dict_20[0][1], rectangle_dict_20[1][1]]]
    else:
        for idx, y in enumerate(y_list):
            if len(y_list) - 2 >= idx >= 1:
                ymin = max(y_list[idx][0], y_list[idx - 1][1])
                y_list[idx][0] = ymin
                y_list[idx - 1][1] = ymin
                ymax = max(y_list[idx][1], y_list[idx + 1][0])
                y_list[idx][1] = ymax
                y_list[idx + 1][0] = ymax
        y_list[-1][1] = rectangle_dict_20_ymax
    detail_len = len(y_list)

    rectangle_list = partition(None, rectangle_dict_20_28, y_list)
    rectangle_list = sum(rectangle_list, [])
    end_index = list(rectangle_dict.keys())[-1]
    append_rectangle_dict = dict(zip(range(end_index + 1, end_index + 1 + len(rectangle_list)), rectangle_list))
    rectangle_dict.update(append_rectangle_dict)

    # detail_dict_20_xy = []
    # x_20_min = rectangle_dict[20][0][0]
    # x_20_max = rectangle_dict[20][1][0]
    # for idx, i in enumerate(y_list):
    #     temp = [[x_20_min, y_list[idx][0]], [x_20_max, y_list[idx][1]]]
    #     detail_dict_20_xy.append(temp)

    return detail_len, end_index, append_rectangle_dict, rectangle_dict



def unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key = lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])

def uncliped_bbox(quadrangle, unclip_ratio, img_height, img_width):
    # quadrangle = convert_coord(bbox)
    quadrangle = unclip(quadrangle, unclip_ratio).reshape(-1, 1, 2)
    quadrangle, sside = get_mini_boxes(quadrangle)
    quadrangle = np.array(quadrangle)
    quadrangle[:, 0] = np.clip(np.round(quadrangle[:, 0]), 0, img_width)
    quadrangle[:, 1] = np.clip(np.round(quadrangle[:, 1]), 0, img_height)
    return quadrangle