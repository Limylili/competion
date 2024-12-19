import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, ROOT
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import pandas as pd

#提供时间戳，以便于测量代码执行的时间
def time_synchronized():
    # 使用 time.time() 代替，确保返回时间戳
    return time.time()

# 进行对象检测的核心函数
def detect(save_img=False):
    # 解析命令行参数：获取模型权重、数据源、图像大小等参数
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # 设置保存图像的标志
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # 判断是否使用摄像头
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    file_name = []
    file_code = []

    # 设置保存目录
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 初始化设置
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # 加载模型权重，获取模型的步长，并调整图像大小以适应模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # 设置数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 从模型中获取类别名称，并为每个类别随机生成颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 运行推理
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # 遍历数据集
    for path, img, im0s, vid_cap in dataset:
        # 图像预处理
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # 应用NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        x_value = {}
        # 用于保存每张图片的检测结果
        for i, det in enumerate(pred):
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 保存检测结果到x_value字典
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        cls = torch.tensor(cls).tolist()
                        x_value[xywh[0]] = int(cls)  # 保存每个检测到的物体的类别和坐标信息

            # 打印推理时间
            print(f'{s}Done. ({t2 - t1:.3f}s)')

        # 保存文件名和文件编码
        file_name.append(os.path.split(path)[-1])
        res = ''
        for key in sorted(x_value):  # 按照xywh顺序排列并生成文件编码
            res += str(x_value[key])  # 获取每个物体的类别信息
        file_code.append(res)  # 添加到file_code

    # 保存检测结果到CSV
    save_csv_path = str(os.getcwd()) + '\\' + str(save_dir) + '\\submission.csv'
    print(f'Submission saved at {save_csv_path}')
    sub = pd.DataFrame({"file_name": file_name, 'file_code': file_code})
    sub.to_csv(save_csv_path, index=False)
    #打印总耗时
    print(f'Done. ({time.time() - t0:.3f}s)')

#主函数
if __name__ == '__main__':
    #解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp17/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'zifu/datasets/images/mchar_test_a', help='source')
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    #检查依赖
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()