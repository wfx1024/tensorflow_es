> 项目地址：https://gitee.com/windandwine/YOLO_v3_tensorflow
> 转载请注明出处

# 一、项目简介

## 1. 检测类别

采用tensorflow（python）实现 YOLO v3目标检测算法，可对图片，包含图片的文件夹、摄像头和视频进行对如下20个类物体的检测。

```json
{"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
 "boat", "traffic", "light", "fire", "hydrant", "stop", "sign", "parking", "meter", 
 "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
 "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", 
 "snowboard", "sports", "ball", "kite", "baseball", "bat", "baseball", "glove", 
 "skateboard", "surfboard", "tennis", "racket", "bottle", "wine", "glass", "cup", 
 "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
 "carrot", "hot", "dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
 "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell", 
 "phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
 "scissors", "teddy", "bear", "hair", "drier", "toothbrush"}
```

## 2.权重文件

模型是基于COCO训练的YOLO v3。

如果你要下载**github**的权重，请点击这里[这里](https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz)

如果你要下载**官网**的权重，请点击这里[这里](https://pjreddie.com/darknet/yolo/)

## 3.文献地址

如果你要查看论文，请点击[这里](https://arxiv.org/abs/1804.02767)

# 二、YOLOv3

## 1.结构

重画了简洁的结构图，图中灰色部分为Darknet-53。代码中封装了yolo_block的结构，如图中粉色所示，yolo_block的输出，一个流向后方，一个是输出，详细内容可以看代码。

![输入图片说明](https://images.gitee.com/uploads/images/2019/0912/212217_8fcdcf83_1295352.png "1567674551447.png")

## 2.模块

其中有两种模块DBL和Resn。

DarkNet-53没有Pooling层和Fc层，而是使用**大步长**（>1）的conv代替pooling（**下采样**），进一步防止有效信息的丢失，这对于小目标来说是十分有利的。

### 1.1DBL

v3也采用BN(batch normalization)，激活函数为**Leaky Relu**，除却个别conv，大部分conv都使用(conv+BN+Leaky Relu)的组合，这也是v3的基本单元，称为DBL。

### 1.2Res

v3是融合了ResNet（残差网络），融合浅层和深层信息，防止有效信息的丢失，同时防止训练深层网络时出现**梯度消失**。Res是ResNet的基本单元。

### 1.3Resn

如图所示，结构为(zero pooling+DBL+Res×n)

## 3.DarkNet-53参数

![输入图片说明](https://images.gitee.com/uploads/images/2019/0912/220139_584512a0_1295352.png "1567566269850.png")

## 4.输出

在v3中，输出三个feature map，大小分别是**(1×13×13×255) (1×26×26×255) (1×52×52×255)**，用于检测不同物体的大小。从255个channel中可以提取出3个anchor boxes，4个pre_boxes，1个置信度confidence，和80个类别概率，
$$
3\times(4+1+80)=255
$$

# 三、运行

项目需要安装tensorflow、opencv-python、numpy。

## 1.测试

预测参数在./setting/predict_args.py中，改成需要的即可。或者直接通过运行脚本输入参数。

目录下有test.py文件，可以做视频或图片的检测。

```python
def main():
    parser = argparse.ArgumentParser(description='YOLO V3 检测文件')
    parser.add_argument('--detect_object', default=detect_object, type=str, help='检测目标-img或video')
    parser.add_argument('--input_image', default=input_image, type=str, help='图片路径')
    parser.add_argument('--input_video', default=input_video, type=str, help='视频路径')
    parser.add_argument('--use_letterbox_resize', type=lambda x: (str(x).lower() == 'true'), default=True, help='是否使用letterbox')

    input_args = parser.parse_args()
    # 图片检测
    if input_args.detect_object == 'img':
        img_origin = cv2.imread(input_args.input_image)  # 原始图片
        if img_origin is None:
            raise Exception('未找到图片文件！')
        img_detect(input_args)

    # 视频检测
    elif input_args.detect_object == 'video':
        vid = cv2.VideoCapture(input_args.input_video)
        if vid is None:
            raise Exception('未找到视频文件!')
        video_detect(input_args)
```

## 2.训练

### 2.1数据集

数据集的文件列表txt记录了图片的信息，包括bbox，类别。每行格式如下

```
line_idx File_name file_w file_h label x1 y1 w1 h1 label x2 y2 w2 h2 ...
```

如有需要可参考./convert/convert_annotation_file.py中更改，将其他格式更改成项目需要的格式。

### 2.2配置

训练参数在./setting/train_args.py下，按需要修改即可。

### 2.3训练

入口func在./train.py中，直接运行即可。

# 五、Tricks

v3很多tricks来自于v2，如BN、multi-scale、特征融合等。

## 1.Batch Normalization

对比v1，v2加入了BN，是mAP提升了2%。

**批量规范化**，简称BN，对数据进行预处理能提升训练速度，提升训练效果，BN基于这个观点，对每一层的输入进行处理。

### 1.1特点

该方法的提出基于以下背景：

1. 神经网络每层输入的分布总是发生变化，通过标准化上层输出，均衡输入数据分布，加快训练速度； 可以设置较大的学习率和衰减，而不用去care初始参数，BN总能快速收敛，调参狗的福音。
2. 通过规范化输入，降低激活函数在特定输入区间达到饱和状态的概率，避免 gradient vanishing 问题；举个例子：0.95^64 ≈ 0.0375    计算累积会产生数据偏离中心，导致误差的放大或缩小。
3. 输入规范化对应样本正则化，在一定程度上可以替代 Drop Out； Drop Out的比例也可以被无视了，全自动的节奏。

### 1.2做法

在**卷积池化之后**，在**激活函数之前**，对每个数据输出进行规范化，均值为0，方差为1。
$$
\begin{aligned} 
    & \color{red}{\widehat{x}^{(k)} =\frac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}} \\
    & \color{red}{y^{(k)} =\gamma^{(k)} \widehat{x}^{(k)}+\beta^{(k)}} \\
\end{aligned}
$$
第一部分是batch内数据归一化，E为batch均值，var为方差，batch数据近似代表整体。

第二部分，引入**附加参数$\gamma$和$\beta$**，分布表示scale和shift。简单的归一化相当于只是用了激活函数的近似线性部分，破坏了原始数据的特征分布，会降低模型表达能力。这两个参数需要训练得到。

![输入图片说明](https://images.gitee.com/uploads/images/2019/0912/215030_9258c8e2_1295352.png "1567321703527.png")

## 2.高分辨率模型

**fine-tuning**：在预训练时，先试用224×224的图片进行训练自己的特征提取网络（160 epochs），这部分称为**分类部分**，然后将输入图片尺度**增加**到448×448，继续使用的检测数据集对其进行fine-tuning（10 epochs），这部分称为**检测部分**。

该方式能提高mAP4%。

## 3.特征提取网络

v1共有98个bbox，用全连接层直接预测bbox坐标，**丢失较多空间信息**，预测不准。

v2将v1的**Fc和最后一个pooling去除**，使得最后的卷积层有更高分辨率的特征，然后缩减网络，**用416×416代替448×448**。


## 4.Anchor boxes

416×416用作输入尺寸最终能输出一个13×13的特征图，特征图有**奇数大小**的宽高，每个特征图在划分单元格（Cell）时只有一个中心单元格（Center Cell）。图片中的物体倾向于出现在中心位置，如果只有一个中心单元格，有利于预测这些物体。

![输入图片说明](https://images.gitee.com/uploads/images/2019/0912/215120_d5197f21_1295352.png "1567344399499.png")

引入Anchor boxes，**预测Anchor boxes的偏移度和置信度，而不是直接预测坐标**，卷积层采用32个值来采样图片，若采用FSRCNN中3组9个的方式，每个Cell可预测出9个，共13×13×9=1521个Anchor Boxes，YOLO v2确定Anchor Boxes的方法是**维度聚类**，**每个Cell选择5个Anchor Box**。

v1为 S²×(B×5+C)=7×7×(2×5+10)，而v2为 S²×K×(5 + C) =13×13×5×(5+20)。

其中K为每个cell预测K（默认K=5）个anchor box，括号中的5是一个置信度4个参数和20个类别概率。

mAP4降低0.2%但recall（查全率）提升7%，feature map分辨率提升，对YOLO本身对小图像检测不足的缺点有所改善。

## 5.维度聚类

Faster R-CNN中Anchor boxes的尺寸是一开始设定的，如果match得好，则结果更准确。

YOLO v2采用K-Means的方式训练anchor boxes，自动找到更好的宽高值用于初始化。

K-means用欧氏距离函数，较大的Anchor Boxes比较小的Anchor Boxes产生更多错误，聚类结果可能会偏离。聚类目的是找到更准确初识Anchor Box参数，与Box大小无关，即提高IOU值，YOLO v2采用IOU值为评判标准，即K-means 采用的距离函数（度量标准） 为：
$$
\color{red}{d(box, centroid)=1-IoU(box, centroid)}
$$
![输入图片说明](https://images.gitee.com/uploads/images/2019/0912/215158_0aa9a563_1295352.png "1567346204082.png")

如图所示，灰色和紫色代表两个不同的数据集。分析聚类结果并权衡模型复杂度与IOU值后，YOLO v2选择K=5，即选择了5种大小的Box维度来进行定位预测。

右侧是预测行人时，初始的anchor box，和手动设置的Anchor Box位置和大小差别显著，**瘦高的框比扁长的框多，位置与中心有偏离**（更符合行人的特征）。

YOLO v2中Cluster IOU生成的5种Anchor的Avg IOU是61，采用9种Anchor Boxes的FSRCNN的Avg IOU是60.9，v2框少但两者成绩相仿，v2性能更好。

![输入图片说明](https://images.gitee.com/uploads/images/2019/0912/215211_564fba3b_1295352.png "1567346695871.png")

## 6.直接位置预测

v2借鉴了Faster R-CNN，但直接对bbox回归可能导致模型不稳定，中心点可能出现在任意位置，导致回归过程振荡。

现在使用正例anchor和GT的预测偏移值（**平移量**和**尺度因子**）如下
$$
\begin{aligned}
& t_{x}=\left(x-x_{a}\right) / w_{a} & t_{y}=\left(x-y_{a}\right) / h_{a} \\
& t_{w}=\log \left(w / w_{a}\right) & t_{h}=\log \left(h / h_{a}\right)\\
\end{aligned}
$$
v2在预测bbox位置参数时采用了强约束方法

![输入图片说明](https://images.gitee.com/uploads/images/2019/0912/215230_ebb70fad_1295352.png "1567400625634.png")

图中黑色虚线框为anchor box，而蓝色框为bbox，预测出的bbox的宽高为(bx,by)和(bw,bh)，计算方式如下
$$
\begin{aligned} 
    & b_x =\sigma(t_x)+c_x \\
    & b_y =\sigma(t_y)+c_y \\
    & b_w =p_w e^{t_w} \\
    & b_h =p_h e^{t_h} \\
\end{aligned}
$$
其中anchor box宽高为$p_w,p_h$，每个bbox有四个参数$(t_x,t_y,t_w,t_h)$，cell大小如图所示$(c_x,c_y)$，$\sigma$为sigmoid激活函数，将函数值约束到(0,1)。

简而言之，$(b_x,b_y)$就是cell附近的anchor box针对预测值$(t_x,t_y)$得到到的 bbox的预测结果，这种方式对于较远距离的bbox预测值$(t_x,t_y)$能够得到很大的限制。

5，6两者结合使mAP提升5%。

## 7.细粒度特征

**Fine-Grained Features（特征融合）**，v2通过添加一个转移层（Passthrough Layer：Route + Reorg），把高分辨率的浅层特征连接到低分辨率的生成特征，堆积在不同channel上，进行融合后检测。

具体操作是先获取前层的26×26的feature map，将其与最后输出的13×13的feature map进行连接，而后输入检测器进行检测，检测器的FC起到了对全局特征融合的作用，以此来提高对小目标的检测能力。

使mAP提升1%。

## 8.multi-Scale

**多尺度训练**，为了适应不同尺度下的检测任务，v2训练网络时，其在检测数据集上fine-tuning时采用的输入图像的size是动态变化的，每训练10个batch，网络随机选择另一种size的输入图像。

v2用参数是32的倍数采样，即采用{320,352,…,608}的输入尺寸（网络会自动改变尺寸，并继续训练的过程）。
![输入图片说明](https://images.gitee.com/uploads/images/2019/0912/215252_78b85f40_1295352.png "1567402058839.png")

这一Trick使得网络在不同输入尺寸上都能达到较好到的预测效果 ，是同一网络在不同分辨率上进行检测。大图慢，小图快，总体提高了准确率。因此多尺度训练算是在准确率和速度上达到一个平衡。

![输入图片说明](https://images.gitee.com/uploads/images/2019/0912/215303_0f7ce528_1295352.png "1567402238405.png")

使mAP提升1%。

