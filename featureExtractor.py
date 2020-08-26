import numpy as np
import cv2
import pywt


def list_patches(img):
    return 0

# convert a list of RBG images/patches into YCrCb channel
def convert_YCrCb(img_list):
    for i in range(img_list.shape[0]):
        img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2YCR_CB)
    return img_list

def sum_var_mean(mat):
    return [np.sum(mat), np.var(mat), np.mean(mat)]

# transform an image/patch into a feature vector
# using 5 level-3 (db1 ... db5) DWt
def daubechies_wavelet_transform(x):
    height, width, n_channel = x.shape
    feature_vector = []
    db_list = ['db1', 'db2', 'db3', 'db4', 'db5']
    level = 3
    for c in range(n_channel):
        for db in db_list:
            coefs_list = pywt.wavedec2(x[:, :, c].reshape((height, width)), db, level=level)
            for i in range(level+1):
                if i == 0:
                    feature_vector.extend(sum_var_mean(coefs_list[0]))
                else:
                    for j in range(3):
                        feature_vector.extend(sum_var_mean(coefs_list[i][j]))
    return np.asarray(feature_vector)
