import torch
import numpy as np
import facenet.utils as utils
from facenet.model import InceptionResnetV1


class FaceNet:
    def __init__(self):
        self.Inet = self.load_inet(
            f_path="./facenet/model_store/vggface2-features.pt",
            l_path="./facenet/model_store/vggface2-logits.pt"
        )

    def load_inet(self, f_path, l_path):
        features_path = f_path
        logits_path = l_path

        state_dict = {}
        state_dict.update(torch.load(features_path, map_location='cpu'))
        state_dict.update(torch.load(logits_path, map_location='cpu'))

        model = InceptionResnetV1()
        model.load_state_dict(state_dict)
        model.eval()

        return model

    @torch.no_grad()
    def FaceCompartor(self, face1, face2):
        face1 = utils.single_img_to_tensor(face1)
        face2 = utils.single_img_to_tensor(face2)

        feature1 = self.Inet(face1).numpy()
        feature2 = self.Inet(face2).numpy()

        dist = np.sqrt(np.sum(np.square(feature1 - feature2)))

        return dist


