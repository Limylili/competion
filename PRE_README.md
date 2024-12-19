# **阿里天池——街景字符编码识别**

## 具体实现流程如下：

### 1、数据准备：
(1)下载官网的mchar_data_list_0515.csv

(2)根据csv中提供的链接，下载官网数据集至mycoco/all_download_data/目录下 

(3)运行merge_images.py文件（将所有数据集存放至mycoco/all_images/）

(4)运行json_to_txt.py文件（制作所有图片对应的txt标签，存放至mycoco/all_labels）

### 2、配置YOLOv5环境
(1)Github下载YOLOv5-6.0源码、YOLOv5s权重文件

(2)根据requirements.txt配置YOLOv5环境

(3)运行make_txt.py文件划分数据集，将all_images中的数据重新划分（将新划分数据集存放至zifu/datasets/）

(4)运行train_val.py文件按照划分好的数据集准备数据

(5)复制配置文件models/yolov5s.yaml至zifu/，修改配置文件中的相关参数

(6)复制配置文件data/coco128.yaml至zifu/，修改配置文件中的相关参数

### 3、训练模型
(1)修改train.py中的相关参数

(2)根据比赛要求修改detect.py文件，需要将预测结果写入csv格式的表格中

### 4、云服务器训练模型

