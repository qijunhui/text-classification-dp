# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/28 18:51
@Author  : qijunhui
@File    : utils.py
"""
import os, csv, requests


def read_csv(filepath, filter_title=False, delimiter=","):
    data = []
    csv.field_size_limit(500 * 1024 * 1024)
    if os.path.exists(filepath):  # 如果目标文件存在:
        with open(filepath, "r") as fr:
            data = csv.reader(fr, delimiter=delimiter)  # 逐行读取csv文件 迭代器变量
            if filter_title:
                next(data)  # 过滤首行
            data = list(data)
        print(f"{filepath} [{len(data)}] 已加载... ")
    else:
        print(f"{filepath} 文件不存在...")
    return data


def save_csv(filepath, data, columns=None, delimiter=","):
    with open(filepath, "w", newline="") as fw:
        csv_writer = csv.writer(fw, delimiter=delimiter)
        if columns:
            csv_writer.writerow(columns)  # 写表头
        csv_writer.writerows(data)
    print(f"{filepath} [{len(data)}] 文件已存储...")


def save_file(filepath, url):
    html = requests.get(url, timeout=10)
    if html.status_code == 200:
        file = html.content  # 字节码 还需要编码
        with open(filepath, "wb") as fw:
            fw.write(file)
    print(f"{filepath} 文件已存储...")
