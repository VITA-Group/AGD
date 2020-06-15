import numpy as np
from skimage.io import imread, imsave
import os
import sys

def color_transfer(content_img, style_img):
    '''
    Transfer style image color to content image.
    Method described in https://arxiv.org/abs/1606.05897
    Args:
        content_img: type=ndarray, shape=(Wc,Hc,C=3)
        style_img: type=ndarray, shape=(Ws,Hs,C=3)
    Returns:
        content_img_hat: content image with the color of style image. type=ndarray, shape=(Wc,Hc,C=3)
    '''

    content_mat = np.transpose(content_img.reshape((-1, content_img.shape[-1]))) # ndarray, shape=(3, W*H)
    style_mat = np.transpose(style_img.reshape((-1, style_img.shape[-1])))

    assert content_mat.shape[0] == 3
    assert style_mat.shape[0] == 3

    # cov matrix:
    content_cov = np.cov(content_mat) # ndarray, shape=(3, 3)
    style_cov = np.cov(style_mat)
    # mean vec:
    content_mean = np.mean(content_mat, axis=-1)
    style_mean = np.mean(style_mat, axis=-1)

    if np.isnan(content_cov).any(): 
        raise ValueError('content_cov as NaN')
    if np.isinf(content_cov).any(): 
        raise ValueError('content_cov as Inf')

    if np.isnan(style_cov).any(): 
        raise ValueError('style_cov as NaN')
    if np.isinf(style_cov).any(): 
        raise ValueError('style_cov as Inf')

    # evd:
    Sc, Uc = np.linalg.eig(content_cov)
    Ss, Us = np.linalg.eig(style_cov)

    content_cov_rec = Uc @ np.diag(Sc) @ Uc.transpose()
    style_cov_rec = Us @ np.diag(Ss) @ Us.transpose()

    assert (Sc>=0).all() # cov matrix should be semi-positive
    assert (Ss>=0).all()

    # linear transform:
    # A = (Us @ np.diag(Ss**0.5)) @ \
    #     (Uc @ np.diag(Sc**(-0.5))).transpose()
    A = (Us @ np.diag(Ss**0.5) @ Us.transpose()) @ \
        (Uc @ np.diag(Sc**(-0.5)) @ Uc.transpose()).transpose()
    b = style_mean - A @ content_mean

    # get new image:
    new_mat = A @ content_mat + np.expand_dims(b, axis=1) # ndarray, shape=(3, W*H)
    content_img_hat = new_mat.transpose().reshape(content_img.shape)

    # deal with image range and dtype:
    content_img_hat[content_img_hat<0] = 0
    content_img_hat[content_img_hat>255] = 255
    content_img_hat = content_img_hat.astype(np.uint8)

    content_hat_cov = np.cov(new_mat)
    content_hat_mean = np.mean(new_mat, axis=-1)

    return content_img_hat

def channel_tranfer(content_img, style_img):
    pass

def color_transfer_per_img(src, dest):
    img1 = imread(src)
    img2 = imread(dest)
    img2_new = color_transfer(img2, img1)
    return img2_new
    # imsave('new_img.png', img2_new)

def post_process_color_transfer(src_folder, dest_folder, save_folder):
    fnames = os.listdir(src_folder)
    for fname in fnames:
        if 'png' in fname:
            assert fname in os.listdir(dest_folder), "File Name Not Matching"

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for fname in fnames:
        if 'png' in fname:
            src = os.path.join(src_folder, fname)
            dest = os.path.join(dest_folder, fname)
            new_img = color_transfer_per_img(src, dest)
            imsave(os.path.join(save_folder, fname), new_img)


if __name__ == "__main__":

    src_folder = sys.argv[1]
    dest_folder = sys.argv[2]
    save_folder = sys.argv[3]

    post_process_color_transfer(src_folder, dest_folder, save_folder)

