from typing import Tuple

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, fill=255, padding_mode="constant")


train_transform = transforms.Compose(
    [
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_transforms(
    phase: str = "all",
) -> Tuple[transforms.Compose, transforms.Compose] | transforms.Compose:
    if phase == "all":
        return train_transform, test_transform
    elif phase == "train":
        return train_transform
    elif phase == "test":
        return test_transform
    else:
        raise ValueError(phase)
