import numpy as np
from skimage.io import imread, imsave
import os
import sys
import matplotlib.pyplot as plt


def draw_hist(fname, save_folder):
    img = imread(fname)
    img_flat = np.reshape(np.array(img),[-1])
    plt.clf()
    plt.hist(img_flat)
    plt.title('Color Distribution Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_folder, os.path.basename(fname)))


if __name__ == "__main__":

    img_folder = sys.argv[1]
    save_folder = sys.argv[2]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    fnames = os.listdir(img_folder)

    for fname in fnames:
        if 'png' in fname:
            draw_hist(os.path.join(img_folder, fname), save_folder)

