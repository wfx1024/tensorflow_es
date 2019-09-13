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


def parse(annotation_file_path, src_img_dir):
    """
    提取每个图片所有标注
    :param annotation_file_path:
    :param src_img_dir:
    :return:
    """
    fr = open(annotation_file_path, 'r')
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


def draw_by_line(line_idx, annotation_file_path):
    """
    读取annotation行，找到对应图片，画框
    :param line_idx:
    :param annotation_file_path:
    :return:
    """
    fr = open(annotation_file_path, 'r')
    lines = fr.readlines()
    this_line = lines[line_idx - 1]
    split_annoatation = this_line.split(" ")
    path = split_annoatation[1]
    img = cv2.imread(path)
    i = 3
    while i + 5 < len(split_annoatation):
        x1, y1, x2, y2 = int(split_annoatation[i + 2]), int(split_annoatation[i + 3]), int(split_annoatation[i + 4]), int(split_annoatation[i + 5])
        i += 5
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('img_detect', img)
    cv2.waitKey(0)


def convert(box):
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


def convert_annotation(annotation_file_path, to_file_path, source_img_dir):
    """
    解析写出（Wider Face数据集标注样式--> 项目可解析的格式）
    :param annotation_file_path: 标注文件路径
    :param to_file_path:
    :param source_img_dir
    :return:
    """
    lines = len(open(annotation_file_path, 'r').readlines())
    fr = open(annotation_file_path, 'r')
    fw = open(to_file_path, 'w')

    img_num = 0
    pbar = tqdm(total=lines)
    for _ in range(lines):
        step = 1  # 进度条步数
        line = fr.readline().rstrip()
        if line:
            num = fr.readline().rstrip()  # bbox数量
            path = line  # 路径
            img_path = source_img_dir + '/' + path
            img = cv2.imread(img_path)
            if img is not None and num is not '0':
                img_num += 1  # 图片数
                shape = img.shape
                fw.write(str(img_num) + ' ' + img_path + ' ')
                fw.write(str(shape[1]) + ' ' + str(shape[0]) + ' ')
                step += 1
                for n in range(int(num)):  # 每个bbox
                    box = fr.readline().rstrip().split()  # 每个bbox
                    if box[2] is not '0' and box[3] is not '0':  # 标注宽或高为0则不写入
                        fw.write('0 ')
                        box = convert(box)
                        step += 1
                        for j in range(4):
                            fw.write(str(box[j]) + ' ')
                fw.write('\n')
            else:
                fr.readline().rstrip()
                step += 1
        pbar.update(step)  # 更新进度条
    print("\033[32m转换完成，共有标注图片{}个".format(img_num))


def main():
    # 训练集
    train_anno_file = "../data/sample/wider_face_train_bbx_gt.txt"  # 注解文件
    train_anno_to_file = '../data/sample/wider_face_train_bbx_gt2.txt'  # 写入文件路径
    train_img_dir = "F:/data_deeplearning/sample_data/WIDERFace/WIDER_train/images"  # 对应的图片文件路径

    # 验证集
    val_anno_file = "../data/sample/wider_face_train_bbx_gt.txt"  # 注解文件
    val_anno_to_file = '../data/sample/wider_face_train_bbx_gt2.txt'  # 写入文件路径
    val_img_dir = "F:/data_deeplearning/sample_data/WIDERFace/WIDER_train/images"  # 对应的图片文件路径

    convert_annotation(val_anno_file, val_anno_to_file, val_img_dir)  # 注解文件转换成可训练的格式

    # parse(val_anno_file, val_img_dir)  # 读取画框所有图像
    # draw_by_line(10966, '../data/sample/wider_face_train_bbx_gt2.txt')


if __name__ == '__main__':
    main()
