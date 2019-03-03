import glob

import numpy as np
from torchvision import transforms
import torch
import torch.utils.data as data
from PIL import Image

class ImageFolder128(Dataset):
    """
    Image shape is (720, 1280, 3) --> (768, 1280, 3) --> 6x10 128x128 patches
    """


    def __init__(self, folder_path):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))


    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = np.array(Image.open(path))
        h, w, c = img.shape

        img = img[100:228,100:228,:]/ 255.0

        img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(img).float()

        img = torch.from_numpy(img).float()

        return img, path


    def get_random(self):
        i = np.random.randint(0, len(self.files))
        return self[i]


    def __len__(self):
        return len(self.files)


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None, loader=default_loader):
        images = []
        for filename in os.listdir(root):
            if is_image_file(filename):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        try:
            img = self.loader(os.path.join(self.root, filename))
        except:
            return torch.zeros((3, 128, 128))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)



