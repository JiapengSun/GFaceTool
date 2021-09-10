import cv2
import torch
import time
import numpy as np
import mtcnn.utils as utils
import core.vision as vision
from mtcnn.models import PNet, RNet, ONet


class MtcnnDetector:
    def __init__(self, threshold):
        self.Pnet = self.load_pnet("./mtcnn/model_store/pnet_epoch.pt")
        self.Rnet = self.load_rnet("./mtcnn/model_store/rnet_epoch.pt")
        self.Onet = self.load_onet("./mtcnn/model_store/onet_epoch.pt")
        self.threshold = threshold

    def load_pnet(self, path):
        pnet = PNet()
        # map_location = lambda storage, loc: storage
        pnet.load_state_dict(torch.load(path, map_location='cpu'))
        pnet.eval()
        return pnet

    def load_rnet(self, path):
        rnet = RNet()
        rnet.load_state_dict(torch.load(path, map_location='cpu'))
        rnet.eval()
        return rnet

    def load_onet(self, path):
        onet = ONet()
        onet.load_state_dict(torch.load(path, map_location='cpu'))
        onet.eval()
        return onet

    def predict_pnet(self, origin_img, pyramids):
        img = origin_img.copy()
        origin_h, origin_w, _ = img.shape

        out = []
        for pyramid in pyramids:
            hs, ws, _ = pyramid
            scale_img = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)
            # cv2.resize方法的参数顺序是 先宽 后高
            input_pnet = utils.single_img_to_tensor(scale_img)
            output_pnet = self.Pnet(input_pnet)
            out.append(output_pnet)
            # 输出的feature_map的顺序是 先高 后宽

        # -------------------------------------#
        #   对pnet处理后的结果进行处理
        # -------------------------------------#
        rectangles = []
        for i in range(len(pyramids)):
            cls_prob = out[i][0][0][0]
            raw_offsets = out[i][1][0]
            rectangle = utils.get_rectangles_pnet(
                cls_prob,
                raw_offsets,
                origin_w,
                origin_h,
                self.threshold[0],
                pyramids[i][2]
            )
            rectangles.extend(rectangle)
            # 此处使用extend是因为rectangle本身就是一个二维list

        rectangles_np = np.stack(rectangles)
        keep = utils.NMS(rectangles_np, 0.7, 'Union')
        rectangles = rectangles_np[keep].tolist()

        return rectangles

    def predict_rnet(self, origin_img, rectangles):
        img = origin_img.copy()
        origin_h, origin_w, _ = img.shape
        predict_24_batch = []

        for rectangle in rectangles:
            crop_img = img[
                       int(rectangle[1]):int(rectangle[3]),
                       int(rectangle[0]):int(rectangle[2])
                       ]
            scale_24_img = cv2.resize(crop_img, (24, 24), interpolation=cv2.INTER_LINEAR)
            predict_24_batch.append(scale_24_img)

        input_rnet = utils.batch_img_to_tensor(predict_24_batch)
        out = self.Rnet(input_rnet)

        cls_prob = np.array(out[0])
        raw_offsets = np.array(out[1])

        rectangles = utils.get_rectangles_rnet(
            cls_prob,
            raw_offsets,
            rectangles,
            origin_w,
            origin_h,
            self.threshold[1]
        )

        return rectangles

    def predict_onet(self, origin_img, rectangles):
        img = origin_img.copy()
        predict_48_batch = []

        for rectangle in rectangles:
            crop_img = img[
                       int(rectangle[1]):int(rectangle[3]),
                       int(rectangle[0]):int(rectangle[2])
                       ]
            scale_48_img = cv2.resize(crop_img, (48, 48), interpolation=cv2.INTER_LINEAR)
            predict_48_batch.append(scale_48_img)

        input_onet = utils.batch_img_to_tensor(predict_48_batch)
        output = self.Onet(input_onet)

        cls_prob = np.array(output[0])
        raw_offsets = np.array(output[1])
        pts_prob = np.array(output[2])

        bboxs, landmarks = utils.get_rectangles_onet(
            cls_prob,
            raw_offsets,
            pts_prob,
            rectangles,
            self.threshold[2]
        )

        return bboxs, landmarks

    @torch.no_grad()
    def FaceDetector(self, img, min_face_size=12, show_steps=False):
        # -------------------------------------#
        #   生成图像金字塔
        # -------------------------------------#
        pyramids = utils.calculate_pyramids(img, min_face_size)
        if show_steps:
            vision.show_pyramids(img, pyramids, is_bgr=True)
        # 输入网络的是BGR图像 用于输出显示的是RGB图像

        # -----------------------------#
        #   粗略计算人脸框 Pnet部分
        # -----------------------------#
        t = time.time()
        rectangles = self.predict_pnet(img, pyramids)
        t1 = time.time() - t
        if show_steps:
            vision.show_boxes(
                origin_img=img,
                bboxes=rectangles,
                show_score=False,
                is_bgr=True,
                subtitle="Result of P-net"
            )

        # -----------------------------#
        #   稍微精确计算人脸框 Rnet部分
        # -----------------------------#
        t = time.time()
        rectangles = self.predict_rnet(img, rectangles)
        t2 = time.time() - t
        if show_steps:
            vision.show_boxes(
                origin_img=img,
                bboxes=rectangles,
                show_score=True,
                is_bgr=True,
                subtitle="Result of R-net"
            )

        # ---------------------------------#
        #   精确计算人脸框与面部特征点 Onet部分
        # ---------------------------------#
        t = time.time()
        bboxs, landmarks = self.predict_onet(img, rectangles)
        t3 = time.time() - t

        if show_steps:
            print(
                "MTCNN time cost " +
                '{:.3f}'.format(t1 + t2 + t3) +
                '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3)
            )

            vision.show_boxes(
                origin_img=img,
                bboxes=bboxs,
                landmarks=landmarks,
                show_score=True,
                is_bgr=True,
                subtitle="Result of O-net"
            )

        return bboxs, landmarks
