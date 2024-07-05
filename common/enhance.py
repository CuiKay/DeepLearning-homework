
from PIL import Image, ImageEnhance
import numpy as np
import cv2

def enhance(img, bri, col, con, sha):
    image = Image.fromarray(img[:,:,::-1]) # bgr -> rgb
    if bri:
        # 亮度增强
        enh_bri = ImageEnhance.Brightness(image)
        brightness = 1.5
        image = enh_bri.enhance(brightness)

    if col:
        # 色度增强
        enh_col = ImageEnhance.Color(image)
        color = 1.5
        image = enh_col.enhance(color)

    if con:
        # 对比度增强
        enh_con = ImageEnhance.Contrast(image)
        contrast = 1.5
        image = enh_con.enhance(contrast)

    if sha:
        # 锐度增强
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = 3.0
        image = enh_sha.enhance(sharpness)

    new_img = np.array(image)[:,:,::-1]
    # cv2.imwrite(r'test/adjust.jpg', new_img)
    return new_img

