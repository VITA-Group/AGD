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
        self._portion = portion

        self.files_A_total = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))

        if self._portion is not None:
            num_files_A = len(self.files_A_total)

            if self._portion > 0:
                split_A = int(np.floor(self._portion * num_files_A))
                self.files_A = self.files_A_total[:split_A]

            elif self._portion < 0:
                split_A = int(np.floor((1 + self._portion) * num_files_A))
                self.files_A = self.files_A_total[split_A:]

        else:
            self.files_A = self.files_A_total


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        return {'A': item_A}

    def __len__(self):
        # return max(len(self.files_A), len(self.files_B))
        return len(self.files_A)