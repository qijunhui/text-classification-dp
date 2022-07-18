# -*- coding: utf-8 -*-
"""
@Time    : 2022/7/18 14:58
@File    : utils.py
@Author  : UU algorithm team
"""
import json

import dill


def get_mean(sets):
    return round(sum(sets) / len(sets), 4) if len(sets) > 0 else 0.0


def save_pkl(filepath, data):
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as fr:
        data = json.load(fp=fr)
    print(f"[{filepath}] [{len(data)}] data saving...")
    return data


def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False)
    print(f"[{filepath}] [{len(data)}] data loading...")
