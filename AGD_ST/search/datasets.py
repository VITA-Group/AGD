import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', portion=None):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self._portion = portion

        self.files_A_total = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.jpg'))
        self.files_B_total = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.jpg'))

        if self._portion is not None:
            num_files_A = len(self.files_A_total)
            num_files_B = len(self.files_B_total)

            if self._portion > 0:
                split_A = int(np.floor(self._portion * num_files_A))
                self.files_A = self.files_A_total[:split_A]

                split_B = int(np.floor(self._portion * num_files_B))
                self.files_B = self.files_B_total[:split_B]   

            elif self._portion < 0:
                split_A = int(np.floor((1 + self._portion) * num_files_A))
                self.files_A = self.files_A_total[split_A:]

                split_B = int(np.floor((1 + self._portion) * num_files_B))
                self.files_B = self.files_B_total[split_B:]

        else:
            self.files_A = self.files_A_total
            self.files_B = self.files_B_total


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        # return max(len(self.files_A), len(self.files_B))
        return len(self.files_A)


class PairedImageDataset(Dataset):
    def __init__(self, dataset_dir, soft_data_dir, mode='train', portion=None, transforms_=None):
        '''
        Construct a dataset with all images from a dir.

        dataset: str. dataset name
        style: str. 'A2B' or 'B2A'
        '''
        self.transform = transforms.Compose(transforms_)
        self._portion = portion
        
        path_A = os.path.join(dataset_dir, '%s/A' % mode)
        path_B = os.path.join(soft_data_dir)
        self.files_A_total = sorted(glob.glob(path_A + '/*.jpg'))
        self.files_B_total = sorted(glob.glob(path_B + '/*.png'))

        assert len(self.files_A_total) == len(self.files_B_total)

        if self._portion is not None:
            num_files = len(self.files_A_total)

            if self._portion > 0:
                split = int(np.floor(self._portion * num_files))
                self.files_A = self.files_A_total[:split]
                self.files_B = self.files_B_total[:split]

            elif self._portion < 0:
                split = int(np.floor((1 + self._portion) * num_files))
                self.files_A = self.files_A_total[split:]
                self.files_B = self.files_B_total[split:]

        else:
            self.files_A = self.files_A_total
            self.files_B = self.files_B_total

        print('files_A:', len(self.files_A))
        print('files_B:', len(self.files_B))


    def __getitem__(self, index):

        if np.random.rand() < 0.5:
            flip = True
        else:
            flip = False

        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_A = img_A.convert("RGB")

        if flip:
            img_A= np.asarray(img_A) # PIL.Image to np.ndarray
            img_A = np.flip(img_A, axis=1) # data augumentation: horrizental flip
            img_A = Image.fromarray(np.uint8(img_A)) # np.ndarray to PIL.Image
            
        item_A = self.transform(img_A)

        img_B = Image.open(self.files_B[index % len(self.files_B)])
        img_B = img_B.convert("RGB")

        if flip:
            img_B= np.asarray(img_B) # PIL.Image to np.ndarray
            img_B = np.flip(img_B, axis=1) # data augumentation: horrizental flip
            img_B = Image.fromarray(np.uint8(img_B)) # np.ndarray to PIL.Image

        item_B = self.transform(img_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return len(self.files_A)