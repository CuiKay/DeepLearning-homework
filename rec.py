
from common.params import args
import cv2
import numpy as np
from ppocr.infer import predict_rec, predict_cls

rec = predict_rec.TextRecognizer(args)
cls = predict_cls.TextClassifier(args)


image_file = r'994ff3aa84eb6fe8a991e72750a8b16.png'

dst_img = cv2.imdecode(np.fromfile(image_file, dtype = np.uint8), cv2.IMREAD_COLOR)
dst_img_list = [dst_img]
text = rec(img_list = dst_img_list)[0][0][0]
print(text)

image_file = r'994ff3aa84eb6fe8a991e72750a8b16.png'
dst_img = cv2.imdecode(np.fromfile(image_file, dtype = np.uint8), cv2.IMREAD_COLOR)
dst_img_list = [dst_img]
angle = cls(img_list = dst_img_list)[1][0][0]
print(angle)