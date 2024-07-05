#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-09 10:49:31
# @Author  : luzhuang (luzhuang@inspur.com)

import os
import json, collections


def jsonfile_to_dict(json_path):
    with open(json_path, 'r', encoding='utf-8') as load_f:
        dict_v = json.load(load_f, object_pairs_hook=collections.OrderedDict)
    return dict_v


def dict_to_jsonfile(dict_v, file_path):
    with open(file_path, "w") as f:
        json.dump(dict_v, f, indent=2, sort_keys=True, ensure_ascii=False)
