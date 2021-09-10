import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet.compartor import FaceNet


if __name__ == '__main__':
    img1 = cv2.imread("demo_img_face/face2.jpg")
    img2 = cv2.imread("demo_img_face/face5.jpg")

    img1 = cv2.resize(img1, (160, 160))
    img2 = cv2.resize(img2, (160, 160))

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fn = FaceNet()
    res = fn.FaceCompartor(img1, img2)

    res_img = np.hstack((img1, img2))
    plt.imshow(res_img)
    plt.text(
        0, 0,
        "Dist {:.5f}".format(res),
        fontdict={'size': 20, 'color': 'red'}
    )
    plt.show()
