
from math import fabs, sin, cos, radians
from common.params import args
from common.ocr_utils import order_points
import cv2
import numpy as np
from math import atan, pi

def get_angle(sta_point, mid_point, end_point):
    ma_x = sta_point[0]-mid_point[0]
    ma_y = sta_point[1]-mid_point[1]
    mb_x = end_point[0]-mid_point[0]
    mb_y = end_point[1]-mid_point[1]
    ab_x = sta_point[0]-end_point[0]
    ab_y = sta_point[1]-end_point[1]
    ab_val2 = ab_x * ab_x + ab_y * ab_y
    ma_val2 = ma_x * ma_x + ma_y * ma_y
    mb_val2 = mb_x * mb_x + mb_y * mb_y
    cos_M = (ma_val2+mb_val2-ab_val2) / (2 * np.sqrt(ma_val2)*np.sqrt(mb_val2))
    angleAMB = np.arccos(cos_M)/np.pi * 180
    return angleAMB

def get_angles(screenCnt, polygon=4):
    if isinstance(screenCnt, list):
        screenCnt_ = screenCnt*2
    else:
        screenCnt_ = screenCnt.tolist()*2
    edge_list = [screenCnt_[i:i+polygon-1] for i in range(4)]
    angle_list = [get_angle(*i) for i in edge_list]
    return angle_list

def initial_verify_corner(screenCnt):
    row1 = (screenCnt[1][1] - screenCnt[0][1]) / (screenCnt[1][0] - screenCnt[0][0])
    row2 = (screenCnt[2][1] - screenCnt[3][1]) / (screenCnt[2][0] - screenCnt[3][0])
    col1 = (screenCnt[3][1] - screenCnt[0][1]) / (screenCnt[3][0] - screenCnt[0][0])
    col2 = (screenCnt[2][1] - screenCnt[1][1]) / (screenCnt[2][0] - screenCnt[1][0])
    degree_row1 = atan(row1) * 180 / pi
    degree_row2 = atan(row2) * 180 / pi
    degree_col1 = atan(col1) * 180 / pi
    degree_col2 = atan(col2) * 180 / pi
    diff_degree_row1 = 0 - degree_row1
    diff_degree_row2 = 0 - degree_row2
    if degree_col1 > 0:
        diff_degree_col1 = 90 - degree_col1
    else:
        diff_degree_col1 = -90 - degree_col1
    if degree_col2 > 0:
        diff_degree_col2 = 90 - degree_col2
    else:
        diff_degree_col2 = -90 - degree_col2
    min_diff_degree_row = min(abs(diff_degree_row1), abs(diff_degree_row2))
    max_diff_degree_row = max(abs(diff_degree_row1), abs(diff_degree_row2))
    min_diff_degree_col = min(abs(diff_degree_col1), abs(diff_degree_col2))
    max_diff_degree_col = max(abs(diff_degree_col1), abs(diff_degree_col2))
    if abs(diff_degree_row1 - diff_degree_row2) > 25 or abs(diff_degree_col1 - diff_degree_col2) > 50:
            screenCnt = None
    if (abs(min_diff_degree_row) < 2 and abs(max_diff_degree_row) > 5) or \
            (abs(min_diff_degree_col) < 2 and abs(max_diff_degree_col) > 25):
        screenCnt = None
    if screenCnt is not None:
        angle_list = get_angles(screenCnt, polygon = 4)
        max_angle, min_angle = max(angle_list), min(angle_list)
        if max_angle - min_angle > 40:
            screenCnt = None
        # dis_90 = [abs(int(90-i)) for i in angle_list]
        # if 1<=dis_90.count(0)<=3:
        #     screenCnt = None
    return screenCnt

def contour_detection(cnt, img):
    screenCnt = None

    cnt = np.int32(cnt).reshape(len(cnt),1,2)
    contourArea = cv2.contourArea(cnt)
    if contourArea / (img.shape[0] * img.shape[1]) < 0.001: # todo
        return screenCnt

    peri = cv2.arcLength(cnt, True)
    for threshold in np.arange(0.025, 0.45, 0.01):
        threshold = round(threshold, 3)
        approx = cv2.approxPolyDP(cnt, threshold * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return screenCnt

    points = sorted(screenCnt.reshape(4,2).tolist(), key = lambda x: x[0])
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

    screenCnt = np.int32([
        points[index_1], points[index_2], points[index_3], points[index_4]
    ])

    if args.is_visualize and screenCnt is not None:
        tmp = img.copy()
        cv2.drawContours(tmp, [screenCnt.reshape(4,-1,2)], -1, (0, 0, 255), 2)
        cv2.imwrite('test/contours.jpg', tmp)
    # screenCnt = order_points(screenCnt.reshape(4, 2))
    screenCnt = screenCnt.reshape(4, 2).astype('float32')
    screenCnt = initial_verify_corner(screenCnt)

    if screenCnt is not None and args.is_visualize:
        temp = img.copy()
        cv2.drawContours(temp, [np.array(screenCnt, dtype = 'int32')], -1, (0, 0, 255), 2)
        for i in screenCnt.tolist():
            cv2.circle(temp, (int(i[0]), int(i[1])), radius = 3, color = (0,0,0), thickness = 3)
        cv2.imwrite('test/contours.jpg', temp)
    return screenCnt


def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img) # 逆时针旋转90度
    return dst_img


def image_location_sort_box(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    pts = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    pts = np.array(pts, dtype="float32")
    # (x1, y1), (x2, y2), (x3, y3), (x4, y4) = _order_points(pts)
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = order_points(pts) #  todo
    return [x1, y1, x2, y2, x3, y3, x4, y4]

def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy

def rotate(x, y, angle, cx, cy):
    angle = angle  # *pi/180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new

def xy_rotate_box(cx, cy, w, h, angle=0, degree=None):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*sin(angle)+cy
    """
    if degree is not None:
        angle = degree
    cx = float(cx)
    cy = float(cy)
    w = float(w)
    h = float(h)
    angle = float(angle)
    x1, y1 = rotate(cx - w / 2, cy - h / 2, angle, cx, cy)
    x2, y2 = rotate(cx + w / 2, cy - h / 2, angle, cx, cy)
    x3, y3 = rotate(cx + w / 2, cy + h / 2, angle, cx, cy)
    x4, y4 = rotate(cx - w / 2, cy + h / 2, angle, cx, cy)
    return x1, y1, x2, y2, x3, y3, x4, y4


def minAreaRect(img, coords):
    rect = cv2.minAreaRect(np.float32(coords))
    points = sorted(list(cv2.boxPoints(rect)), key = lambda x: x[0])
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
    # screenCnt = initial_verify_corner(np.float32(box))
    screenCnt = np.float32(box)

    if screenCnt is not None and args.is_visualize:
        temp = img.copy()
        cv2.drawContours(temp, [np.array(screenCnt, dtype = 'int32')], -1, (0, 0, 255), 2)
        for i in screenCnt.tolist():
            cv2.circle(temp, (int(i[0]), int(i[1])), radius = 3, color = (0,0,0), thickness = 3)
        cv2.imwrite('test/contours.jpg', temp)
    return screenCnt