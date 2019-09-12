> 项目地址：https://gitee.com/windandwine/YOLO_v3_tensorflow
> 转载请注明出处

# 一、项目介绍


## 1. 介绍
采用tensorflow（python）实现 YOLO v1目标检测算法，可对图片，包含图片的文件夹、摄像头和视频进行对如下20个类物体的检测。

```json
{"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"}
     
```

![输入图片说明](https://images.gitee.com/uploads/images/2019/0903/173939_47c6ac87_1295352.jpeg "person.jpg")

![输入图片说明](https://images.gitee.com/uploads/images/2019/0903/173954_c5509657_1295352.jpeg "person.jpg")

## 2.使用说明

在项目下yolo.py文件中，main方法含有四种检测方式，注释掉其他三种即可。

```python
if __name__ == '__main__':
    yolo = Yolo()
    # 图片检测
    yolo.image_detector('test/sample/person.jpg', 'test/result/person.jpg', 'test/result/person.txt')
    # 多图检测
    # yolo.multi_img_detector('test/sample', 'test/result')
    # # 摄像头检测
    # yolo.camera_detector('test/result/camera.mp4')
    # # 视频检测
    # yolo.vedio_detector('test/sample/india_street.mp4', 'test/result/india_street.mp4')
```

其中除camera_detector只有**保存文件**一个参数，其他三个方法都需要**加载文件**和**保存文件**两个参数。

## 3.权重文件

如果你要下载**百度网盘**的权重，请点击这里[这里](https://pan.baidu.com/s/1i57uPLF)

如果你要下载**谷歌云**的权重，请点击这里[这里](https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing)

## 4.文献地址

如果你要查看论文，请点击[这里](https://arxiv.org/abs/1506.02640)

# 二、YOLO v1

## 1.候选区域

### 1.1置信度
![输入图片说明](https://images.gitee.com/uploads/images/2019/0903/171315_61f9d368_1295352.png "1567153281362.png")

YOLO将图片分成S×S的网格，每个单元格检测中心点在该网格的目标，每个单元格会生成**B**个以其为中心的**锚点框（anchor box）**，anchor box有四个参数(x,y,w,h)，分别是**中心坐标和宽高**，每个anchor box有一个**置信度（confidence score）**，某个anchor box的置信度如下
$$
\color{red}{C=P(Object)\times IoU}
$$
第1项为**框内含有目标的概率**，记为P(object)，包含目标时P(object)=1，没有目标（背景）时P(object)=0。

第2项为**候选框的准确度**，用anchor box和ground truth（实际框）的**交并比（IoU）**来表示。

### 1.2交并比

$$
\color{red}{IoU(R,R')=\frac{R\cap R'}{R\cup R'}}
$$

![IoU](https://images.gitee.com/uploads/images/2019/0903/171341_f9b2a912_1295352.png "IoU")

### 1.3坐标系

左上角为原点(0, 0)，坐标数值如下图。anchor box参数x,y的单位是**一个网格的长度宽度**而不是像素，w,h为相对原图的大小，取值[0,1]。

最终输出包含5个元素[x,y,w,h,C]。

![输入图片说明](https://images.gitee.com/uploads/images/2019/0903/171438_3a0c2885_1295352.png "1567134423844.png")

## 2.分类

### 2.1概率

任务需要预测n种的物体。每个单元的anchor box内存在目标时，是某个类别（$class_i$）的条件概率为$P(class_i|Object)$。不管一个单元格预测了多少个anchor boxes，只输出一组类别概率。
$$
\left[\begin{array}{c}
P(class_1|Object)\\
P(class_2|Object)\\
\vdots\\
P(class_n|Object)\\
\end{array}\right]
$$

### 2.2参数个数

每个单元格需要输出 B×5+n个参数，总体需要输出S²×(B×5+C)大小的张量。

对于PASAL VOC数据，共有20个类别（n=20），将图片分割成7×7（S=7），如果每个单元格预测2个anchor box（B=2），最终的张量大小为
$$
7\times 7 \times30 (2\times5+20)=7\times 7 \times30
$$
![输入图片说明](https://images.gitee.com/uploads/images/2019/0903/171503_cba6c8c4_1295352.png "1567143285258.png")
## 3.损失函数

![输入图片说明](https://images.gitee.com/uploads/images/2019/0903/172346_09a1f0bf_1295352.png "图像 16.png")

## 4.网络

### 4.1特征提取网络

特征提取网络参考GoogleNet模型，包含**24个卷积层和2个全连接层**，采用Leaky ReLU作为激活函数，最后一层为线性激活。（也可以替换为VGG-16，代码里就是VGG16）。

![输入图片说明](https://images.gitee.com/uploads/images/2019/0903/171525_c5a14378_1295352.png "1567143364009.png")

### 4.2训练网络

![输入图片说明](https://images.gitee.com/uploads/images/2019/0903/171539_28ebc481_1295352.png "1567147305428.png")

最后输出7×7×30的张量，一个点的深度为30，如下

![输入图片说明](https://images.gitee.com/uploads/images/2019/0903/171556_0b8214fd_1295352.png "1567146057236.png")

### 4.3筛选候选框

使用非极大值抑制。