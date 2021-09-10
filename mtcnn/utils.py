import math
import torch
import numpy as np
import torchvision.transforms as transforms


# -------------------------------------#
#   计算图像金字塔参数
# -------------------------------------#
def calculate_pyramids(img, min_face_size):
    factor = 0.709
    net_size = 12

    # image.shape 方法返回的顺序是 先高 后宽
    height, width, _ = img.shape

    # 根据最小人脸尺寸与网络大小进行第一次resize
    cur_scale = float(net_size) / min_face_size
    cur_width = math.ceil((width * cur_scale))
    cur_height = math.ceil((height * cur_scale))

    # 计算所有可能的金字塔尺度
    scales = []
    while min(cur_width, cur_height) >= net_size:
        # ensure width and height are even
        w = cur_width
        h = cur_height
        scales.append((h, w, cur_scale))

        cur_scale *= factor
        cur_width = math.ceil(cur_width * factor)
        cur_height = math.ceil(cur_height * factor)

    return scales


# -------------------------------------#
#   将图片转换为Tensor
# -------------------------------------#
def single_img_to_tensor(img):
    # Pnet的输入必须进行归一化
    transform = transforms.ToTensor()
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    return img


# -------------------------------------#
#   将pnet输出的人脸batch转换为Tensor
# -------------------------------------#
def batch_img_to_tensor(img_batch):
    transform = transforms.ToTensor()
    trans_batch = []
    for img in img_batch:
        trans_img = transform(img)
        trans_batch.append(trans_img)

    trans_batch = torch.stack(trans_batch)

    return trans_batch


# -------------------------------------#
#   对输出的框进行最终确认 防止出现错误
# -------------------------------------#
def pick_rectangles(rectangles, width, height):
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return pick


# -------------------------------------------#
#   将长方形调整为正方形并扩大范围便于后续旋转处理
# -------------------------------------------#
def rectangle_to_square(rectangles, scale=1.0):
    w = rectangles[:, 2] - rectangles[:, 0] + 1
    h = rectangles[:, 3] - rectangles[:, 1] + 1
    l = np.maximum(w, h).T
    l = l * scale
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T - 1
    return rectangles


# -------------------------------------#
#   非极大抑制
# -------------------------------------#
def NMS(rectangles, threshold, mode="Union"):
    x1 = rectangles[:, 0]
    y1 = rectangles[:, 1]
    x2 = rectangles[:, 2]
    y2 = rectangles[:, 3]
    scores = rectangles[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 如果填写了不存在的模式那么默认按Union处理

        index = np.where(ovr <= threshold)[0]
        order = order[index + 1]

    return keep


# -------------------------------------#
#   对pnet处理后的结果进行处理
# -------------------------------------#
def get_rectangles_pnet(cls_prob, raw_offsets, width, height, threshold, scale):
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    raw_offsets = np.swapaxes(raw_offsets, 1, 2)

    stride = 2
    cell_size = 12

    (x, y) = np.where(cls_prob >= threshold)

    score = np.array((cls_prob[x, y]))
    score = score[:, np.newaxis]

    bbox = np.array([x, y]).T
    bb1 = np.fix((stride * bbox + 0) / scale)
    bb2 = np.fix((stride * bbox + cell_size) / scale)
    bbox = np.concatenate((bb1, bb2), axis=1)

    dx1, dy1, dx2, dy2 = [raw_offsets[j, x, y] for j in range(4)]
    offsets = np.array(torch.stack([dx1, dy1, dx2, dy2], 1))

    bbox = bbox + offsets * 12.0 / scale

    rectangles = np.concatenate((bbox, score), axis=1)
    rectangles = rectangle_to_square(rectangles)

    keep = NMS(rectangles, 0.5, 'Union')
    keep = rectangles[keep].tolist()
    pick = pick_rectangles(keep, width, height)

    return pick


# -------------------------------------#
#   对rnet处理后的结果进行处理
# -------------------------------------#
def get_rectangles_rnet(cls_prob, raw_offsets, rectangles, width, height, threshold):
    prob = cls_prob[:, 0]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    sc = np.array([prob[pick]]).T
    x1, y1, x2, y2 = [rectangles[pick, i] for i in range(4)]
    dx1, dy1, dx2, dy2 = [raw_offsets[pick, j] for j in range(4)]

    w = x2 - x1
    h = y2 - y1

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dy1 * h)[0]]).T
    x2 = np.array([(x2 + dx2 * w)[0]]).T
    y2 = np.array([(y2 + dy2 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)

    rectangles = rectangle_to_square(rectangles)

    keep = NMS(rectangles, 0.7, 'Union')
    keep = rectangles[keep].tolist()
    pick = pick_rectangles(keep, width, height)

    return pick


# -------------------------------------#
#   对onet处理后的结果进行处理
# -------------------------------------#
def get_rectangles_onet(cls_prob, raw_offsets, pts, rectangles, threshold):
    prob = cls_prob[:, 0]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    sc = np.array([prob[pick]]).T
    x1, y1, x2, y2 = [rectangles[pick, i] for i in range(4)]
    dx1, dy1, dx2, dy2 = [raw_offsets[pick, j] for j in range(4)]

    w = x2 - x1
    h = y2 - y1

    pts_x = []
    for i in range(0, 10, 2):
        pts_temp = np.array([(w * pts[pick, i] + x1)[0]]).T
        pts_x.append(pts_temp)

    pts_y = []
    for i in range(1, 10, 2):
        pts_temp = np.array([(w * pts[pick, i] + y1)[0]]).T
        pts_y.append(pts_temp)

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dy1 * h)[0]]).T
    x2 = np.array([(x2 + dx2 * w)[0]]).T
    y2 = np.array([(y2 + dy2 * h)[0]]).T

    rectangles = np.concatenate(
        (x1, y1, x2, y2, sc,
         pts_x[0], pts_y[0], pts_x[1], pts_y[1], pts_x[2],
         pts_y[2], pts_x[3], pts_y[3], pts_x[4], pts_y[4]), axis=1)

    keep = NMS(rectangles, 0.7, 'Minimum')
    bbox = rectangles[keep, 0:5]
    bbox = rectangle_to_square(bbox, scale=1.2).tolist()
    landmarks = rectangles[keep, 5:15].tolist()

    return bbox, landmarks
