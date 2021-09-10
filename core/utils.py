import cv2
import math
import core.vision as vision


def get_aligned_faces(origin_img, bboxs, landmarks, is_bgr=False, show_steps=False):
    img = origin_img.copy()
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 第一步 裁剪出单张人脸
    croped_faces = []
    croped_landmarks = []
    for i, box in enumerate(bboxs):
        # 截取单张人脸图像
        # 检查box的位置防止出现越过图片边界的情况
        box = check_border(img, box)
        crop_face = img[
                    int(box[1]):int(box[3]),
                    int(box[0]):int(box[2])
                    ]

        # 取得单张人脸的特征点坐标并进行转换
        crop_landmark = landmarks[i]
        for j in range(0, 10, 2):
            crop_landmark[j] = crop_landmark[j] - box[0]
            crop_landmark[j + 1] = crop_landmark[j + 1] - box[1]

        croped_faces.append(crop_face)
        croped_landmarks.append(crop_landmark)

    if show_steps:
        vision.show_faces(
            faces=croped_faces,
            landmarks=croped_landmarks,
            subtitle="Detected Faces"
        )

    # 第二步 根据特征点进行对齐
    rotated_faces = []
    rotated_landmarks = []
    for i, crop_face in enumerate(croped_faces):
        landmark = croped_landmarks[i]

        rotate_face, rotate_landmark = align_face(crop_face, landmark)
        rotated_faces.append(rotate_face)
        rotated_landmarks.append(rotate_landmark)

    if show_steps:
        vision.show_faces(
            faces=rotated_faces,
            landmarks=rotated_landmarks,
            subtitle="Faces after Rotation"
        )

    # 第三步 将对齐后的人脸进行二次裁剪
    re_croped_faces = []
    re_croped_landmarks = []
    for i, rotated_face in enumerate(rotated_faces):
        rotated_landmark = rotated_landmarks[i]

        re_crop_face, re_crop_landmark = re_crop_face_landmark(rotated_face, rotated_landmark)
        re_croped_faces.append(re_crop_face)
        re_croped_landmarks.append(re_crop_landmark)

    if show_steps:
        vision.show_faces(
            faces=re_croped_faces,
            landmarks=re_croped_landmarks,
            subtitle="Faces after Re-crop"
        )

    return re_croped_faces, re_croped_landmarks


def align_face(face, landmark):
    left_eye_x = landmark[0]
    left_eye_y = landmark[1]
    right_eye_x = landmark[2]
    right_eye_y = landmark[3]

    eye_center = ((left_eye_x + right_eye_x) / 2,
                  (left_eye_y + right_eye_y) / 2)

    dy = right_eye_y - left_eye_y
    dx = right_eye_x - left_eye_x

    angle = math.atan2(dy, dx) * 180.0 / math.pi

    rotate_matrix = cv2.getRotationMatrix2D(
        eye_center,
        angle,
        scale=1
    )
    rotate_face = cv2.warpAffine(
        face,
        rotate_matrix,
        (face.shape[1], face.shape[0])
    )

    # 对所有特征点进行旋转
    rotate_landmark = landmark.copy()
    for i in range(0, 10, 2):
        x, y = rotate_point(
            eye_center,
            (landmark[i], landmark[i+1]),
            angle,
            face.shape[0]
        )
        rotate_landmark[i] = x
        rotate_landmark[i+1] = y

    return rotate_face, rotate_landmark


def rotate_point(center, point, angle, row):
    x1, y1 = point
    x2, y2 = center
    y1 = row - y1
    y2 = row - y2

    angle = math.radians(angle)
    # angle = math.pi / 180.0 * angle

    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y

    return x, y


def re_crop_face_landmark(rotated_face, rotated_landmark):
    re_crop_landmark = rotated_landmark.copy()

    left_eye = (rotated_landmark[0], rotated_landmark[1])
    right_mouth = (rotated_landmark[8], rotated_landmark[9])
    nose = (rotated_landmark[4], rotated_landmark[5])

    x = -1 * nose[0]
    y = -1 * nose[1]
    for i in range(0, 10, 2):
        x += rotated_landmark[i]
        y += rotated_landmark[i+1]
    x /= 4
    y /= 4
    center = (x, y)

    len_x = left_eye[0] - right_mouth[0]
    len_y = left_eye[1] - right_mouth[1]
    length = 1.25 * math.sqrt(len_x**2 + len_y**2)

    left_top = (center[0] - length, center[1] - length)

    new_box = [
        center[0] - length,
        center[1] - length,
        center[0] + length,
        center[1] + length
    ]

    new_box = check_border(rotated_face, new_box)

    re_crop_face = rotated_face[
        int(new_box[1]):int(new_box[3]),
        int(new_box[0]):int(new_box[2])
    ]

    for i in range(0, 10, 2):
        re_crop_landmark[i] = re_crop_landmark[i] - left_top[0]
        re_crop_landmark[i+1] = re_crop_landmark[i+1] - left_top[1]

    return re_crop_face, re_crop_landmark


def face_resize_160(faces):
    resized_faces = []
    for face in faces:
        face = cv2.resize(face, (160, 160))
        resized_faces.append(face)

    return resized_faces


# -------------------------------------#
#   对bbox进行检查防止出现越过图像边界的情况
# -------------------------------------#
def check_border(origin_img, origin_box):
    box = origin_box.copy()

    right_top_y = 0
    right_top_x = 0
    left_bottom_y = origin_img.shape[0]
    left_bottom_x = origin_img.shape[1]

    if box[1] < right_top_y:
        box[1] = right_top_y
    if box[3] > left_bottom_y:
        box[3] = left_bottom_y
    if box[0] < right_top_x:
        box[0] = right_top_x
    if box[2] > left_bottom_x:
        box[2] = left_bottom_x

    return box
