import torch
import torch.nn.functional as F
import torchvision
import cv2
import math
from pathlib import Path
import argparse
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

# 立体相机参数配置类
class StereoCameraConfig:
    def __init__(self):
        # 左右相机的内参矩阵和畸变系数
        self.cam_matrix_left = np.array([[431.34, 0, 301.14],
                                         [0, 428.26, 241.95],
                                         [0, 0, 1]], dtype=np.float64)
        self.distortion_l = np.array([0.1, -0.1, 0, 0], dtype=np.float64)

        self.cam_matrix_right = np.array([[434.22, 0, 296.92],
                                          [0, 431.88, 241.79],
                                          [0, 0, 1]], dtype=np.float64)
        self.distortion_r = np.array([0.1, -0.1, 0, 0], dtype=np.float64)

        # 立体相机的旋转矩阵和平移向量 (由标定工具获取)
        self.R = np.eye(3, dtype=np.float64)  # 旋转矩阵
        self.T = np.array([-120, 0, 0], dtype=np.float64).reshape(3, 1)  # 平移向量，基线12cm对应-120mm
        self.B = np.abs(self.T[0, 0]) / 1000.0  # 基线长度，单位：米
        self.fx = self.cam_matrix_left[0, 0]

# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=0
    )

    # 生成校正映射
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q

# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2

# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output

# 视差图空洞填充
def insertDepth32f(depth):
    height, width = depth.shape
    integralMap = np.zeros((height, width), dtype=np.float64)
    ptsMap = np.zeros((height, width), dtype=np.int32)

    # 计算积分图
    integralMap[depth > 1e-3] = depth[depth > 1e-3]
    ptsMap[depth > 1e-3] = 1

    # 计算积分图累加
    for i in range(height):
        for j in range(1, width):
            integralMap[i, j] += integralMap[i, j - 1]
            ptsMap[i, j] += ptsMap[i, j - 1]

    for i in range(1, height):
        for j in range(width):
            integralMap[i, j] += integralMap[i - 1, j]
            ptsMap[i, j] += ptsMap[i - 1, j]

    dWnd = 2
    while dWnd > 1:
        wnd = int(dWnd)
        dWnd /= 2
        for i in range(height):
            for j in range(width):
                left = max(j - wnd - 1, 0)
                right = min(j + wnd, width - 1)
                top = max(i - wnd - 1, 0)
                bot = min(i + wnd, height - 1)

                ptsCnt = ptsMap[bot, right] + ptsMap[top, left] - (ptsMap[top, right] + ptsMap[bot, left])
                sumGray = integralMap[bot, right] + integralMap[top, left] - (
                            integralMap[top, right] + integralMap[bot, left])

                if ptsCnt > 0:
                    depth[i, j] = sumGray / ptsCnt

        s = max(wnd // 2 * 2 + 1, 3)
        if s > 201:
            s = 201
        depth = cv2.GaussianBlur(depth, (s, s), s, s)

    return depth

# YOLO相关函数

def load_model(weights, device=None):
    ckpt = torch.load(str(weights), map_location=device)
    ckpt = ckpt['model'].to(device).float()
    return ckpt.eval()

def letterbox(im, new_shape=(640, 640), stride=32, color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    dw, dh = new_shape[1] - int(shape[1] * r), new_shape[0] - int(shape[0] * r)
    dw /= 2
    dh /= 2
    if shape[::-1] != (int(shape[1] * r), int(shape[0] * r)):
        im = cv2.resize(im, (int(shape[1] * r), int(shape[0] * r)), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=300, nm=0):
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 7680
    max_nms = 30000
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]
        conf, j = x[:, 5:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        else:
            x = x[x[:, 4].argsort(descending=True)]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]
    return output

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    c, mh, mw = protos.shape
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih
    masks = crop_mask(masks, downsampled_bboxes)
    if upsample:
        masks = F.interpolate(masks[None], (ih, iw), mode='bilinear', align_corners=False)[0]
    return masks.gt_(0.5)

def crop_mask(masks, boxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
        pad = ((im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2)
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))  # 确保尺寸匹配 (宽, 高)
    if len(masks.shape) == 2:
        masks = masks[:, :, np.newaxis]
    return masks

def run(weights=ROOT / 'weights/yolov5s-seg.pt',
        imgsz=[640, 640],
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        agnostic_nms=False,
        fp16=False):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #model = load_model(weights, device)
    weights_path = r'E:\yolo\runs\train-seg\exp9\weights\best.pt'  # 训练好的模型路径
    model = load_model(weights_path, device)
    #print(model)
    names = model.names
    model.half() if fp16 else model.float()

    # 初始化立体相机配置
    config = StereoCameraConfig()
    height, width = 480, 640  # 左右图像的高度和宽度
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)

    # 打开双目摄像头（分辨率1280x480，左图640x480，右图640x480）
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 初始化立体匹配对象
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 12,  # 必须是16的倍数
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=16,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    sift = cv2.SIFT_create()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 分割左右图像
        left_img = frame[:, :640, :]
        right_img = frame[:, 640:, :]

        # 畸变校正和立体校正
        left_rectified, right_rectified = rectifyImage(left_img, right_img, map1x, map1y, map2x, map2y)

        # 计算视差图
        gray_left_rectified = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right_rectified = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(gray_left_rectified, gray_right_rectified).astype(np.float32) / 16.0

        # 填充视差图中的空洞
        filled_disparity = insertDepth32f(disparity)

        # 归一化视差图便于可视化
        disparity_visual = cv2.normalize(filled_disparity, None, 0, 255, cv2.NORM_MINMAX)
        disparity_visual = np.uint8(disparity_visual)

        # YOLO推理仅在左图进行
        im0 = left_rectified.copy()
        im, ratio, (dw, dh) = letterbox(left_rectified, imgsz, stride=32)
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.half() if fp16 else im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]
        pred, proto = model(im)[:2]
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic_nms, max_det=max_det, nm=32)

        # 假设pred为推理结果
        conf_thres = 0.65  # 置信度阈值
        # 遍历每个图像的检测结果
        for i in range(len(pred)):
            if len(pred[i]) > 0:
                # 保留置信度大于conf_thres的物体
                pred[i] = pred[i][pred[i][:, 4] > conf_thres]

        # 若成功计算视差图，则在掩码位置替换为深度图
        for i, det in enumerate(pred):
            if len(det):
                # 处理mask
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
                masks_ = masks.permute(1, 2, 0).contiguous().cpu().numpy()  # HWC
                new_mask = scale_image(im.shape[2:], (masks_ * 255).astype(np.uint8), im0.shape)

                # 确保 new_mask 具有三个维度
                if len(new_mask.shape) == 2:
                    new_mask = new_mask[:, :, np.newaxis]

                # new_mask 为多个obj的mask通道，可对每个mask区域用 disparity_visual 替换
                # 这里假设 disparity_visual 和 im0 大小一致
                if disparity_visual is not None and disparity_visual.shape[:2] == im0.shape[:2]:
                    # 将掩码区域替换为深度图区域 (深度图为单通道，转换为 BGR 显示)
                    depth_bgr = cv2.cvtColor(disparity_visual, cv2.COLOR_GRAY2BGR)
                    # 对每个 mask 通道进行合成
                    for m_i in range(new_mask.shape[2]):
                        mask_bin = new_mask[:, :, m_i]
                        # mask_bin 是单通道的 0/255 图，将其转换为 bool
                        mask_bool = mask_bin > 128
                        im0[mask_bool] = depth_bgr[mask_bool]

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # 绘制目标框和标签
                for *xyxy, conf, cls in reversed(det[:, :6]):
                    c = int(cls)
                    # 初始化标签为类别名称和置信度
                    label = f'{names[c]} {conf:.2f}'

                    # 计算距离
                    x1, y1, x2, y2 = map(int, xyxy)
                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, filled_disparity.shape[1] - 1))
                    x2 = max(0, min(x2, filled_disparity.shape[1] - 1))
                    y1 = max(0, min(y1, filled_disparity.shape[0] - 1))
                    y2 = max(0, min(y2, filled_disparity.shape[0] - 1))

                    # 提取边界框内的视差值
                    disparity_roi = filled_disparity[y1:y2, x1:x2]
                    # 过滤掉无效视差值
                    valid_disparity = disparity_roi[disparity_roi > 0]
                    if valid_disparity.size > 0:
                        # 使用中位数视差以减少噪声影响
                        disparity_median = np.median(valid_disparity)
                        # 计算深度 Z = (f * B) / disparity
                        Z = (config.fx * config.B) / disparity_median

                        #ground_truth_depth=0.4
                        # 假设获取的深度Z为估计值，ground_truth_depth为真实深度
                        #depth_error = np.abs(Z - ground_truth_depth)  # 绝对误差
                        #depth_error_percentage = (depth_error / ground_truth_depth) * 100  # 误差百分比
                        #print(depth_error_percentage)
                        # 将距离转换为米，并保留两位小数
                        label += f' {Z:.2f}m'
                    else:
                        label += ' N/A'

                    p1, p2 = (x1, y1), (x2, y2)
                    cv2.rectangle(im0, p1, p2, (255, 255, 0), 2, lineType=cv2.LINE_AA)

                    # 计算中心坐标
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # 绘制中心
                    cv2.circle(im0, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)

                    # 计算文本大小
                    w, h = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                    # 确定文本放置位置，放在重心旁边
                    text_x = cx + 10  # 向右偏移10像素
                    text_y = cy
                    if text_x + w > im0.shape[1]:
                        text_x = cx - w - 10  # 如果超出右边界，则放在左边
                    if text_y - h < 0:
                        text_y = cy + h + 10  # 如果超出上边界，则放在下方

                    cv2.putText(im0, label, (text_x, text_y),
                                0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    #cv2.putText(im0, f"Depth Accuracy: {depth_error_percentage:.2f}m", (50, 50),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('YOLO+Depth', im0)
        #cv2.imshow('left_img', left_img)
        #cv2.imshow('right_img', right_img)
        #cv2.imshow('Disparity Map', disparity_visual)

        # 按 "q" 或 "ESC" 退出
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main(opt):
    run(**vars(opt))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device')
    parser.add_argument('--fp16', type=bool, default=False)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

