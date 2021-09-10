import torch
import torchvision.transforms as transforms


def single_img_to_tensor(img):
    transform = transforms.ToTensor()
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    return img
