
from common.ocr_utils import args
from PIL import Image
from math import fabs, sin, cos, radians
import numpy as np
import cv2, time, ast
import pandas as pd
from common.ocr_utils import order_points, fourxy2twoxy

def get_img_rot_broa(img, degree=90):
    b, g, r = list(map(int, cv2.meanStdDev(img)[0].squeeze()))
    borderValue = (b, g, r)
    height, width = img.shape[:2]
    height_new = int(width * fabs(sin(radians(degree))) +
                     height * fabs(cos(radians(degree))))
    width_new = int(height * fabs(sin(radians(degree))) +
                    width * fabs(cos(radians(degree))))
    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    mat_rotation[0, 2] += (width_new - width) / 2
    mat_rotation[1, 2] += (height_new - height) / 2
    img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new),
                                 borderValue=borderValue)
    # mask = np.zeros((height_new + 2, width_new + 2), np.uint8)
    # mask[:] = 0
    # seed_points = [(0, 0), (0, height_new - 1), (width_new - 1, 0),
    #                (width_new - 1, height_new - 1)]
    # for i in seed_points:
    #     cv2.floodFill(img_rotated, mask, i, (255, 255, 255))
    return img_rotated


def text_area(img, bbox_list):
    bbox_list = np.array(bbox_list)
    top_left = int(bbox_list[:,0].min()), int(bbox_list[:,1].min())
    bottm_right = int(bbox_list[:,2].max()), int(bbox_list[:,3].max())
    x_min = max(0, top_left[0]-100)
    x_max = min(img.shape[1], bottm_right[0]+100)
    y_min = max(0, top_left[1]-100)
    y_max = min(img.shape[0], bottm_right[1]+100)
    if (y_max - y_min) / img.shape[0] < 0.8:
        crop_img = img[y_min:y_max, :]
    else:
        crop_img = img
    if (x_max - x_min) / img.shape[1] < 0.8:
        crop_img = crop_img[:, x_min:x_max]
    if args.is_visualize:
        cv2.imwrite(r'test/adjust.jpg', crop_img)
    return crop_img

def text_image_direction(img, orientaion_model, adjust=True):
    def cross_cut(img, n=2):
        size_img = img.shape[:2]
        weight = int(size_img[1] // n)
        height = int(size_img[0] // n)
        img_list = []
        for j in range(n):
            for k in range(n):
                y_min, y_max = height * j, height * (j + 1)
                x_min, x_max = weight * k, weight * (k + 1)
                box_img = img[y_min:y_max, x_min:x_max]
                img_list.append(box_img)
        return img_list

    h, w = img.shape[:2]
    if adjust:
        thesh = 0.1
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[ymin:ymax, xmin:xmax]
    input_data = cross_cut(img[:, :, ::-1], n = 2)
    angle_result = orientaion_model.predict(input_data = input_data)
    angle_result = sum(list(angle_result), [])
    angle_result = sorted(angle_result, key = lambda x: x['scores'][0], reverse = True)
    angle_result_ = [i for i in angle_result if i['scores'][0] > 0.5]
    if len(angle_result_) >= 1:
        angle_result = angle_result_
    angle_result = [i['label_names'][0] for i in angle_result]
    angle = pd.DataFrame(angle_result).mode().values.tolist()[0][0]
    angle = ast.literal_eval(angle)
    return angle

