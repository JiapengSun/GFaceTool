import cv2
import core.utils as utils
from mtcnn.detector import MtcnnDetector


if __name__ == '__main__':
    img = cv2.imread("demo_img_group/group6.jpg")
    # 由cv2读取得到的是BRG图像
    # 输入网络的是BGR图像
    # 用于输出显示效果的是RGB图像

    detector = MtcnnDetector(threshold=[0.6, 0.7, 0.8])

    bboxs, landmarks = detector.FaceDetector(
        img=img,
        min_face_size=16,
        show_steps=True
    )

    aligned_faces, aligned_landmarks = utils.get_aligned_faces(
        origin_img=img,
        bboxs=bboxs,
        landmarks=landmarks,
        is_bgr=True,
        show_steps=False
    )

