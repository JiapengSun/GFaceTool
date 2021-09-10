import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# -------------------------------------#
#   显示金字塔
# -------------------------------------#
def show_pyramids(origin_img, pyramids, is_bgr=False):
    img = origin_img.copy()
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pyramid = np.asarray(img)

    for h, w, f in pyramids:
        im = cv2.resize(img, (w, h))
        # cv2.resize方法的参数顺序是 先宽 后高
        im = np.asarray(im)
        img_pyramid[0:h, 0:w] = im
        # 从np数组中读取图片的时候的顺序是 先高 后宽

    plt.imshow(img_pyramid)
    plt.suptitle('Image Pyramids', fontsize=20)
    plt.show()


# -------------------------------------#
#   显示图片中的人脸框
# -------------------------------------#
def show_boxes(origin_img, bboxes, landmarks=None, show_score=False, is_bgr=False, subtitle=None):
    img = origin_img.copy()
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)

    for box in bboxes:
        width = box[2] - box[0] + 1
        height = box[3] - box[1] + 1
        length = max(width, height)
        rect = patches.Rectangle(
            (box[0], box[1]),
            width,
            height,
            linewidth=1.5,
            edgecolor='green',
            facecolor='none'
        )
        ax.add_patch(rect)

        if show_score:
            score = round(box[4], 2)
            plt.text(
                box[0], box[1],
                score,
                color='yellow',
                size=10
            )

    if landmarks is not None:
        for landmark in landmarks:
            for i in range(0, 10, 2):
                cir = patches.Circle(
                    (landmark[i], landmark[i + 1]),
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none',
                    radius=3
                )
                ax.add_patch(cir)
    plt.suptitle(subtitle, fontsize=20)
    plt.show()


# -------------------------------------#
#   显示提取到的人脸
# -------------------------------------#
def show_faces(faces, landmarks=None, subtitle=None):
    nrow = int(math.sqrt(len(faces)))
    ncol = math.ceil(len(faces) / nrow)

    fig, ax = plt.subplots(nrow, ncol)

    for i, face in enumerate(faces):
        if nrow == 1 and ncol == 1:
            cur_ax = ax
        elif nrow == 1 and ncol > 1:
            cur_ax = ax[i]
        else:
            row = int(i / ncol)
            col = int(i % ncol)
            cur_ax = ax[row][col]

        cur_ax.axis('off')
        cur_ax.imshow(face)
        cur_ax.text(
            0, 0,
            face.shape[0],
            color='black',
            size=10
        )

        if landmarks is not None:
            landmark = landmarks[i]
            for j in range(0, 10, 2):
                cir = patches.Circle(
                    (landmark[j], landmark[j + 1]),
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none',
                    radius=face.shape[0]/40
                )
                cur_ax.add_patch(cir)

    plt.suptitle(subtitle, fontsize=14)
    plt.show()
