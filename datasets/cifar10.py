from typing import Optional, Callable, Tuple, Any

import numpy as np

import torchvision
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms



class CroppedCIFAR10(CIFAR10):

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(
            self,
            crop_width: int,
            crop_height: int,
            root: str = "./data",
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CroppedCIFAR10, self).__init__(
            root,
            transform=None,
            target_transform=target_transform,
            train=train,
            download=download
        )
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.input_transform = transform


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)

        w, h = img.size

        if self.crop_width < w or self.crop_height < h:
            left = np.random.randint(0, w - self.crop_width)
            right = left + self.crop_width

            upper = np.random.randint(0, h - self.crop_height)
            lower = upper + self.crop_height

            img_crop = img.crop([left, upper, right, lower])

        else:
            img_crop = img

        if self.input_transform is not None:
            img_crop = self.input_transform(img_crop)
            img = self.input_transform(img)


        return img_crop, img, target


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    bs = 4
    dta = CroppedCIFAR10(root="./data/cifar10", crop_width=16, crop_height=16, transform=transform)
    dtaloader = torch.utils.data.DataLoader(dta, batch_size=bs, shuffle=False)
    dataiter = iter(dtaloader)
    crops, img, labels = next(dataiter)
    # print labels
    print(' '.join(f'{CroppedCIFAR10.classes[labels[j]]:5s}' for j in range(bs)))
    from img_utils import imshow
    print()
    imshow(torchvision.utils.make_grid(crops))
    imshow(torchvision.utils.make_grid(img))