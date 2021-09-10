from mtcnn.detector import MtcnnDetector
from facenet.compartor import FaceNet
import core.vision as vision
import core.utils as utils
import numpy as np
import time
import cv2
import os


class Gface:
    def __init__(self):
        self.detector = MtcnnDetector(threshold=[0.6, 0.7, 0.8])
        self.compartor = FaceNet()

    def GetFace(self, candidate_image, min_face_size=28, show_steps=False):
        bboxs, landmarks = self.detector.FaceDetector(
            img=candidate_image,
            min_face_size=min_face_size,
            show_steps=show_steps
        )

        aligned_face, _ = utils.get_aligned_faces(
            origin_img=candidate_image,
            bboxs=bboxs,
            landmarks=landmarks,
            is_bgr=True,
            show_steps=show_steps
        )

        resized_face = utils.face_resize_160(aligned_face)

        return resized_face

    def UpdateFace(self, face_id, face_file_name, show_steps=False):
        face_img = cv2.imread("./temp_img/" + face_file_name)
        # 如果没有读取到照片则直接结束并返回 false
        if face_img is None:
            return False

        aligned_face = self.GetFace(
            candidate_image=face_img,
            min_face_size=128,
            show_steps=show_steps
        )[0]

        # aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)

        save_path = "./core/facedata/" + face_id + ".jpg"
        cv2.imwrite(save_path, aligned_face)

        return True

    def DetectWithList(self, id_list, photo_file_name, show_steps=False):
        photo_img = cv2.imread("./temp_img/" + photo_file_name)

        # 根据ID从数据库中提取人脸
        nodata_id_list = []
        saved_id_list = []
        saved_face_list = []
        for faceid in id_list:
            read_path = "./core/facedata/" + faceid + ".jpg"
            read_face = cv2.imread(read_path)

            if read_face is not None:
                saved_id_list.append(faceid)
                saved_face_list.append(read_face)
            else:
                nodata_id_list.append(faceid)

        # 从合照中获取人脸
        candidate_face_list = self.GetFace(
            candidate_image=photo_img,
            min_face_size=18,
            show_steps=show_steps
        )

        # 遍历数据库中的人脸和图像获取的人脸进行对比
        # 将ID分类为found与unfound
        t0 = time.time()
        found_id_list = []
        unfound_id_list = []
        compare_face_list = []
        for i, saved_face in enumerate(saved_face_list):
            min_dist = 1.0
            min_loc = -1
            for j, candidate_face in enumerate(candidate_face_list):
                dist = self.compartor.FaceCompartor(saved_face, candidate_face)
                # 获取欧氏距离最小的人脸的索引
                if dist < min_dist:
                    min_dist = dist
                    min_loc = j

            if min_loc != -1:
                # 按需展示匹配情况
                if show_steps:
                    cp_face = np.hstack((saved_face, candidate_face_list[min_loc]))
                    compare_face_list.append(cp_face)
                # 将已经被识别到的人脸从数据库列表中删除防止重复匹配
                del candidate_face_list[min_loc]
                found_id_list.append(saved_id_list[i])
            else:
                unfound_id_list.append(saved_id_list[i])

        if show_steps:
            t1 = time.time() - t0
            print("FaceNet time cost {:.3f} ".format(t1))
            vision.show_faces(
                faces=compare_face_list,
                landmarks=None,
                subtitle="Data Face and Detected Face"
            )

        return found_id_list, unfound_id_list, nodata_id_list

    def DetectWithoutList(self, photo_file_name, show_steps=False):
        data_path = "./core/facedata/"
        id_list = []
        for cur_file in (os.listdir(data_path)):
            id_list.append(cur_file.strip(".jpg"))

        found_id_list, _, _ = self.DetectWithList(
            id_list=id_list,
            photo_file_name=photo_file_name,
            show_steps=show_steps
        )

        return found_id_list
