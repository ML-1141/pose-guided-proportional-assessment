import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class SiamesePoseDatasetSK(Dataset):
    """
    Siamese dataset using only the given sk paths.

    class_to_imgs: dict {class_name -> [img_paths_for_this_split]}
    pos_ratio: probability that a sampled pair is positive (same class).
    """
    def __init__(self, class_to_imgs, transform=None, pos_ratio=0.5):
        self.transform  = transform
        self.pos_ratio  = pos_ratio

        # keep only classes that actually have images
        self.class_to_imgs = {
            cls: paths for cls, paths in class_to_imgs.items() if len(paths) > 0
        }
        self.classes     = sorted(self.class_to_imgs.keys())
        self.num_classes = len(self.classes)

        # flat list: (img_path, class_idx)
        self.samples = []
        for cls_idx, cls_name in enumerate(self.classes):
            for p in self.class_to_imgs[cls_name]:
                self.samples.append((p, cls_idx))

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        img_path, cls_idx = self.samples[idx]

        # decide positive or negative pair
        make_positive = random.random() < self.pos_ratio

        if make_positive:
            # same class (positive)
            same_class_imgs = self.class_to_imgs[self.classes[cls_idx]]
            img2_path = random.choice(same_class_imgs)
            label = 1
        else:
            # different class (negative)
            other_classes = list(range(self.num_classes))
            other_classes.remove(cls_idx)
            neg_cls_idx = random.choice(other_classes)
            neg_cls_name = self.classes[neg_cls_idx]
            img2_path = random.choice(self.class_to_imgs[neg_cls_name])
            label = -1


        img1 = self._load_img(img_path)
        img2 = self._load_img(img2_path)
        label = torch.tensor(label, dtype=torch.long)

        return img1, img2, label
