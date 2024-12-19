import os
import cv2
import json

# 设置图像和标注文件的路径
train_image_path = './mchar_train/'#下载好的数据集位置
val_image_path = './mchar_val/'
train_annotation_path = './mchar_train.json'
val_annotation_path = './mchar_val.json'

# 读取训练集和验证集的标注数据
train_data = json.load(open(train_annotation_path))
val_data = json.load(open(val_annotation_path))

# 创建存放转换后的标签文件
label_path = '../all_labels/'
if not os.path.exists(label_path):
    os.makedirs(label_path)

# 遍历训练集的数据，并将标注信息转换为YOLO格式
for key in train_data:
    f = open(label_path+key.replace('.png', '.txt'), 'w') #为每个图像文件创建一个文本文件，用于保存YOLO格式的标签
    img = cv2.imread(train_image_path+key) #使用OpenCV的imread函数读取图像文件
    shape = img.shape #获取图像的形状
    label = train_data[key]['label'] #类别标签
    left = train_data[key]['left'] #边界框左上角坐标
    top = train_data[key]['top']
    height = train_data[key]['height'] #边界框高度
    width = train_data[key]['width'] #边界框宽度
    #遍历当前图像的所有标注对象
    for i in range(len(label)):
        #计算每个标注对象的中心点坐标（x_center, y_center）和宽高（w, h）
        x_center = 1.0 * (left[i]+width[i]/2) / shape[1]
        y_center = 1.0 * (top[i]+height[i]/2) / shape[0]
        #归一化到0和1之间
        w = 1.0 * width[i] / shape[1]
        h = 1.0 * height[i] / shape[0]
        #将类别标签和归一化后的边界框坐标写入到之前创建的文本文件中
        f.write(str(label[i]) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + '\n')
    #关闭文本文件，完成当前图像的标签写入
    f.close()

# 遍历验证集的数据，并将标注信息转换为YOLO格式
for key in val_data:
    f = open(label_path+'val_'+key.replace('.png', '.txt'), 'w')
    img = cv2.imread(val_image_path+key)
    shape = img.shape
    label = val_data[key]['label']
    left = val_data[key]['left']
    top = val_data[key]['top']
    height = val_data[key]['height']
    width = val_data[key]['width']
    for i in range(len(label)):
        x_center = 1.0 * (left[i]+width[i]/2) / shape[1]
        y_center = 1.0 * (top[i]+height[i]/2) / shape[0]
        w = 1.0 * width[i] / shape[1]
        h = 1.0 * height[i] / shape[0]
        # label, x_center, y_center, w, h
        f.write(str(label[i]) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + '\n')
    f.close()