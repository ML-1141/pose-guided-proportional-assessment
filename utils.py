import torch
import torch.nn.functional as F

from model import SiameseSTN_AMCNN

@torch.no_grad()
def evaluate_similarity(model: SiameseSTN_AMCNN, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    model.eval()
    z1, z2 = model(x1, x2)
    return F.cosine_similarity(z1, z2, dim=1)


import torchvision.transforms.functional as TF
from PIL import Image

class ResizeWithPad:
    """
    Resize image so its longer side = target_size,
    keep aspect ratio, then pad to (target_size, target_size)
    """
    def __init__(self, target_size=156, fill=0):
        self.target_size = target_size
        self.fill = fill  # padding color

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)

        # resize while keeping aspect ratio
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, (new_h, new_w))

        # compute padding amounts
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h

        left   = pad_w // 2
        right  = pad_w - left
        top    = pad_h // 2
        bottom = pad_h - top

        img = TF.pad(img, (left, top, right, bottom), fill=self.fill)
        return img

