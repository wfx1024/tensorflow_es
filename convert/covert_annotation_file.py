# coding: utf-8
import os
import cv2

"""
修改Wider Face数据集注解文件格式，并保存
"""

"""
原格式:
File name
Number of bounding box
x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

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
File name x1, y1, w, h, 21,  x1, y1, w, h,21 ...
21是face类别
"""


def draw(image_list, src_img_dir=None, tar_img_dir=None):
    if not os.path.exists(tar_img_dir):
        os.mkdir(tar_img_dir)
    for item in image_list:
        sub_path = item["path"]
        path_seg = sub_path.split("/")
        # path = os.path.join(src_img_dir, sub_path)
        path = src_img_dir + '/' + sub_path
        boxes = item["boxes"]
        img = cv2.imread(path)
        # cv2.imshow('sd', img)
        # cv2.waitKey(0)
        for box in boxes:
            ord = box.split(" ")
            x, y, w, h = int(ord[0]), int(ord[1]), int(ord[2]), int(ord[3])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('sd', img)
        cv2.waitKey(0)
        tar_dir = os.path.join(tar_img_dir, path_seg[0])
        if not os.path.exists(tar_dir):
            os.mkdir(tar_dir)
        tar_path = os.path.join(tar_dir, path_seg[1])
        cv2.imwrite(tar_path, img)


def parse(label_file_path, src_img_dir, tar_img_dir):
    fr = open(label_file_path, 'r')
    image_list = []
    line = fr.readline().rstrip()
    # while line:
    mdict = {}
    path = line
    mdict["path"] = path
    num = fr.readline().rstrip()
    boxes_list = []
    for n in range(int(num)):
        box = fr.readline().rstrip()
        boxes_list.append(box)
    mdict["boxes"] = boxes_list
    image_list.append(mdict)
    # line = fr.readline().rstrip()
    draw(image_list, src_img_dir, tar_img_dir)


if __name__ == '__main__':
    file_path = "F:/sample_data2/WIDERFace/wider_face_split/wider_face_train_bbx_gt.txt"
    source_img_dir = "F:/sample_data2/WIDERFace/WIDER_train/images"
    target_img_dir = "data"
    parse(file_path, source_img_dir, target_img_dir)
