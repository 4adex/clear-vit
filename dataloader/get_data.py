import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageEnhance
import json
import os


num_epochs = 90
batch_size = 256


class RandAugment:
    def __init__(self, n=9, m=0.5):
        self.n = n
        self.m = m  # [0, 30] in paper, but we use [0, 1] for simplicity
        self.augment_list = [
            self.auto_contrast, self.equalize, self.rotate, self.solarize, 
            self.color, self.contrast, self.brightness, self.sharpness,
            self.shear_x, self.shear_y, self.translate_x, self.translate_y,
            self.posterize, self.solarize_add, self.invert, self.identity
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(img)
        return img

    def auto_contrast(self, img):
        return ImageOps.autocontrast(img)

    def equalize(self, img):
        return ImageOps.equalize(img)

    def rotate(self, img):
        return TF.rotate(img, self.m * 30)

    def solarize(self, img):
        return TF.solarize(img, int((1 - self.m) * 255))

    def color(self, img):
        return TF.adjust_saturation(img, 1 + self.m)

    def contrast(self, img):
        return TF.adjust_contrast(img, 1 + self.m)

    def brightness(self, img):
        return TF.adjust_brightness(img, 1 + self.m)

    def sharpness(self, img):
        return ImageEnhance.Sharpness(img).enhance(1 + self.m)

    def shear_x(self, img):
        return TF.affine(img, 0, [0, 0], 1, [self.m, 0])

    def shear_y(self, img):
        return TF.affine(img, 0, [0, 0], 1, [0, self.m])

    def translate_x(self, img):
        return TF.affine(img, 0, [int(self.m * img.size[0] / 3), 0], 1, [0, 0])

    def translate_y(self, img):
        return TF.affine(img, 0, [0, int(self.m * img.size[1] / 3)], 1, [0, 0])

    def posterize(self, img):
        return TF.posterize(img, int((1 - self.m) * 8))

    def solarize_add(self, img):
        return TF.solarize(TF.adjust_brightness(img, 1 + self.m), int((1 - self.m) * 255))

    def invert(self, img):
        return TF.invert(img) if random.random() < 0.5 else img

    def identity(self, img):
        return img

class Mixup(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, batch):
        images, labels = batch
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]
        return mixed_images, labels_a, labels_b, lam

class CutMix(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, batch):
        images, labels = batch
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, _, H, W = images.shape
        cx = np.random.uniform(0, W)
        cy = np.random.uniform(0, H)
        w = W * np.sqrt(1 - lam)
        h = H * np.sqrt(1 - lam)
        x0 = int(np.clip(cx - w // 2, 0, W))
        y0 = int(np.clip(cy - h // 2, 0, H))
        x1 = int(np.clip(cx + w // 2, 0, W))
        y1 = int(np.clip(cy + h // 2, 0, H))
        index = torch.randperm(batch_size)
        images[:, :, y0:y1, x0:x1] = images[index, :, y0:y1, x0:x1]
        lam = 1 - ((x1 - x0) * (y1 - y0) / (W * H))
        labels_a, labels_b = labels, labels[index]
        return images, labels_a, labels_b, lam

class RandomErasing(nn.Module):
    def __init__(self, probability=0.25, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
        super().__init__()
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def forward(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, self.r2)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                else:
                    img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                return img
        return img

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = nn.functional.log_softmax(pred, dim=1)
        return torch.mean(torch.sum(-smooth_one_hot * log_prob, dim=1))
    

# Updated ImageNet100Dataset
class ImageNet100Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dirs, labels_file, transform=None, augment=None):
        self.transform = transform
        self.augment = augment
        self.images = []
        self.labels = []
        self.label_to_idx = {}
        
        with open(labels_file, 'r') as f:
            label_dict = json.load(f)
        
        unique_labels = sorted(label_dict.keys())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        for root_dir in root_dirs:
            for label in os.listdir(root_dir):
                label_path = os.path.join(root_dir, label)
                if os.path.isdir(label_path):
                    for img_name in os.listdir(label_path):
                        img_path = os.path.join(label_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.augment:
            image = self.augment(image)
        
        label = torch.tensor(label)
        
        return image, label

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    RandAugment(n=3, m=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomErasing(probability=0.25)
])

val_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom collate function for Mixup and CutMix
def collate_fn(batch):
    images, labels = torch.utils.data.default_collate(batch)
    if random.random() < 0.5:
        return Mixup(alpha=0.8)((images, labels))
    else:
        return CutMix(alpha=1.0)((images, labels))


# Create the datasets
train_dirs = [
    '/home/hawkeye/works/vit/data/train.X1',
    '/home/hawkeye/works/vit/data/train.X2',
    '/home/hawkeye/works/vit/data/train.X3',
    '/home/hawkeye/works/vit/data/train.X4'
]
val_dir = ['/home/hawkeye/works/vit/data/val.X']
labels_file = '/home/hawkeye/works/vit/data/Labels.json'

train_dataset = ImageNet100Dataset(
    root_dirs=train_dirs,
    labels_file=labels_file,
    transform=train_transform
)

val_dataset = ImageNet100Dataset(
    root_dirs=val_dir,
    labels_file=labels_file,
    transform=val_transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)





