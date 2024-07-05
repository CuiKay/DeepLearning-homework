# -*- coding:utf-8 -*-
import requests, json
from common.ocr_utils import imagefile_to_string

if __name__ == '__main__':
    url = f'http://127.0.0.1:5000/kuaidi_rec'
    # url = f'https://1z41330c01.hsk.top/kuaidi_rec'

    filename = r'test/1/微信图片_20240218143429.jpg'

    img_str = imagefile_to_string(filename)
    data = {"imgBase64": img_str}
    headers = {'Content-Type': 'application/json'}
    r = requests.post(url = url, headers = headers, data = json.dumps(data))
    result = r.json()
    print(result)

