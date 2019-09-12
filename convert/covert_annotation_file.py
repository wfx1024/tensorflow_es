# coding: utf-8
import os
import cv2
from tqdm import trange, tqdm
import time

"""
修改Wider Face数据集注解文件格式，并保存
"""

"""
原格式:
File name
Number of bounding box
x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
x1, y1为左上坐标, w,h为宽高

blur:
  clear->0
  normal blur->1
  heavy blur->2

expression:
  typical expression->0
  exaggerate expression->1

illumination:
  normal illumination->0
  extreme illumination->1

occlusion:
  no occlusion->0
  partial occlusion->1
  heavy occlusion->2

pose:
  typical pose->0
  atypical pose->1

invalid:
  false->0(valid image)
  true->1(invalid image)
  
例子：
0--Parade/0_Parade_marchingband_1_849.jpg
1
449 330 122 149 0 0 0 0 0 0 
0--Parade/0_Parade_Parade_0_904.jpg
1
361 98 263 339 0 0 0 0 0 0 
0--Parade/0_Parade_marchingband_1_799.jpg
2
78 221 7 8 2 0 0 0 0 0 
78 238 14 17 2 0 0 0 0 0 
"""

"""
现格式:
line_idx File_name file_w file_h label x1 y1 w1 h1 label x2 y2 w2 h2 ...
label是类别index
"""


def draw(annotation, src_img_dir=None):
    """
    画框并展示
    :param annotation:
    :param src_img_dir:
    :return:
    """

    sub_path = annotation["path"]
    path = src_img_dir + '/' + sub_path
    boxes = annotation["boxes"]
    img = cv2.imread(path)
    for box in boxes:
        ord = box.split(" ")
        x, y, w, h = int(ord[0]), int(ord[1]), int(ord[2]), int(ord[3])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('img_detect', img)


def parse(label_file_path, src_img_dir):
    """
    提取每个图片所有标注
    :param label_file_path:
    :param src_img_dir:
    :return:
    """
    fr = open(label_file_path, 'r')
    line = fr.readline().rstrip()
    while line:
        annotation = {}
        path = line
        annotation["path"] = path
        num = fr.readline().rstrip()
        boxes_list = []
        for n in range(int(num)):
            box = fr.readline().rstrip()
            boxes_list.append(box)
        annotation["boxes"] = boxes_list
        draw(annotation, src_img_dir)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
        line = fr.readline().rstrip()


def convert1(img_size, box):
    """
    左上角坐标，改成中点，并归一化
    :param img_size: 图片大小
    :param box: x, y, w, h
    :return:
    """
    dw = 1. / (img_size[0])
    dh = 1. / (img_size[1])

    # (x + y) / 2
    x = float(box[0]) + float(box[2]) / 2.
    y = float(box[1]) + float(box[3]) / 2.
    w = float(box[2])
    h = float(box[3])

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [round(x, 5), round(y, 5), round(w, 5), round(h, 5)]


def convert2(box):
    """
    宽高改成右下坐标
    :param box: x, y, w, h
    :return:
    """
    x_max = int(box[0]) + int(box[2])
    y_max = int(box[1]) + int(box[3])
    box[2] = str(x_max)
    box[3] = str(y_max)
    return box


def convert_annotation(label_file_path, to_file_path, source_img_dir):
    """
    解析写出
    :param label_file_path:
    :param to_file_path:
    :param source_img_dir
    :return:
    """
    lines = len(open(label_file_path, 'r').readlines())
    fr = open(label_file_path, 'r')
    fw = open(to_file_path, 'w')

    img_num = 0
    pbar = tqdm(total=lines)
    for _ in range(lines):
        step = 1  # 进度条步数
        line = fr.readline().rstrip()
        if line:
            img_num += 1  # 图片数
            path = line  # 路径
            img_path = source_img_dir + '/' + path
            img = cv2.imread(img_path)
            shape = img.shape
            fw.write(str(img_num) + ' ' + img_path + ' ')
            fw.write(str(shape[1]) + ' ' + str(shape[0]) + ' ')
            num = fr.readline().rstrip()  # bbox数量
            step += 1
            for n in range(int(num)):  # 每个bbox
                fw.write('0 ')
                box = fr.readline().rstrip().split()  # 每个bbox
                box = convert2(box)
                step += 1
                for j in range(4):
                    fw.write(str(box[j]) + ' ')
            fw.write('\n')
        pbar.update(step)  # 更新进度条
    print("\033[32m转换完成，共有标注图片{}个".format(img_num))


def main():
    file_path = "../data/sample/wider_face_val_bbx_gt.txt"  # 注解文件
    source_img_dir = "F:/sample_data2/WIDERFace/WIDER_val/images"  # 对应的图片文件路径
    to_file_dir = '../data/sample/wider_face_val_bbx_gt3.txt'  # 写入文件路径
    # parse(file_path, source_img_dir)  # 读取画框所有图像
    convert_annotation(file_path, to_file_dir, source_img_dir)  # 转换convert


if __name__ == '__main__':
    main()
