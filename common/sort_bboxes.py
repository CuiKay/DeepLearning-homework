#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2022/11/02
# @Author  : liulei
# @File    :
# @Software: PyCharm
# @Introduction:

import numpy as np

def coord_convert(bboxes):
    # 4 points coord to 2 points coord for rectangle bbox
    x_min, y_min, x_max, y_max = \
        min(bboxes[0::2]), min(bboxes[1::2]), max(bboxes[0::2]), max(bboxes[1::2])
    return [x_min, y_min, x_max, y_max]

def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    if a_y_min >= b_y_min and a_y_max <= b_y_max:  # todo
        return True

    if b_y_min >= a_y_min and b_y_max <= a_y_max:  # todo
        return True

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

def flatten(sorted_groups, sorted_bbox_groups):
    idxs = []
    bboxes = []
    for group, bbox_group in zip(sorted_groups, sorted_bbox_groups):
        for g, bg in zip(group, bbox_group):
            idxs.append(g)
            bboxes.append(bg.tolist())
    return idxs, bboxes

def sort_line_bbox(g, bg):
    # todo 修改，避免sorted_xs中x值相同
    xs = [bg_item[0] for bg_item in bg]
    xs = [str(index) + '_' + str(i) for index, i in enumerate(xs)]
    sorted_xs = sorted(xs, key = lambda x: float(x.split('_')[1]))

    g_sorted = [None]*len(xs)
    bg_sorted = [None]*len(xs)
    for index, (g_item, bg_item) in enumerate(zip(g, bg)):
        idx = sorted_xs.index(str(index) + '_' + str(bg_item[0]))
        bg_sorted[idx] = bg_item
        g_sorted[idx] = g_item

    return g_sorted, bg_sorted

def sort_bbox(end2end_xyxy_bboxes, min_y_overlap_ratio = 0.2):
    groups = []
    bbox_groups = []
    for index, end2end_xywh_bbox in enumerate(end2end_xyxy_bboxes):
        this_bbox = end2end_xywh_bbox
        if len(groups)==0: #  todo
            groups.append([index])
            bbox_groups.append([this_bbox])
        else:
            flag = False
            for g, bg in zip(groups, bbox_groups):
                if is_on_same_line(this_bbox, bg[0], min_y_overlap_ratio = min_y_overlap_ratio):
                    g.append(index)
                    bg.append(this_bbox)
                    flag = True
                    break
            if not flag:
                groups.append([index])
                bbox_groups.append([this_bbox])

    tmp_groups, tmp_bbox_groups = [], []
    for g, bg in zip(groups, bbox_groups):
        g_sorted, bg_sorted = sort_line_bbox(g, bg)
        tmp_groups.append(g_sorted)
        tmp_bbox_groups.append(bg_sorted)

    # todo 修改，避免sorted_ys中y值相同
    sorted_groups = [None]*len(tmp_groups)
    sorted_bbox_groups = [None]*len(tmp_bbox_groups)
    ys = [bg[0][1] for bg in tmp_bbox_groups]
    ys = [str(index) + '_' + str(i) for index, i in enumerate(ys)]
    sorted_ys = sorted(ys, key = lambda x: float(x.split('_')[1]))
    for index, (g, bg) in enumerate(zip(tmp_groups, tmp_bbox_groups)):
        idx = sorted_ys.index(str(index) + '_'+ str(bg[0][1]))
        sorted_groups[idx] = g
        sorted_bbox_groups[idx] = bg

    end2end_sorted_idx_list, end2end_sorted_bbox_list = flatten(sorted_groups, sorted_bbox_groups)
    return end2end_sorted_bbox_list


if __name__ == "__main__":
    end2end_xyxy_bboxes = np.array([[200,200,300,200,300,400,200,400],
                                    [0,0,100,100,100,200,0,200],
                                    [50,50,100,100,100,150,50,150]
                                    ])
    result = sort_bbox(end2end_xyxy_bboxes)
    print(result)