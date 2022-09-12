import os
from torch.utils.data import Dataset
import torch
from glob import glob
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
    def __init__(self,root,transform=None,loader=default_loader):
        # super(ImageFolder,self).__init__()
        images=[]
        for filename in os.listdir(root):
            if is_image_file(filename):
                images.append('{}' .format(filename))

        self.root=root
        self.transform=transform
        self.images=images
        self.loader=loader

    def __getitem__(self,index):
        filename=self.images[index]
        img=self.loader(os.path.join(self.root,filename))

        if self.transform is not None:
            img=self.transform(img)
        return img

    def __len__(self):
        return len(self.images)


class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)

