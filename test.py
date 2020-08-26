import numpy as np
import os
import cv2
import random
import pywt
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
from scipy.io import loadmat
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load('trained_MLP_sklearn.sav')

# load mean and var for normalization
mean_and_var_dict = loadmat('preprocessed_data/mean_and_var_of_trainset.csv')
mean = mean_and_var_dict['mean']
var = mean_and_var_dict['var']

data = loadmat('preprocessed_data/train.csv')
X_train = data['feature vector']
N = X_train.shape[0]
dim = X_train.shape[1]
y_train = data['label'].reshape(N)

for i in range(N):
    X_train[i] = (X_train[i] - mean) / var

pred = model.predict(X_train)
print(accuracy_score(pred, y_train))

# -----------------------------------------------------------------
"""
x = pywt.data.camera().astype(np.float32)
shape = x.shape

max_lev = 3       # how many levels of decomposition to draw
label_levels = 3  # how many levels to explicitly label on the plots

fig, axes = plt.subplots(2, 4, figsize=[14, 8])
for level in range(0, max_lev + 1):
    if level == 0:
        # show the original image before decomposition
        axes[0, 0].set_axis_off()
        axes[1, 0].imshow(x, cmap=plt.cm.gray)
        axes[1, 0].set_title('Image')
        axes[1, 0].set_axis_off()
        continue

    # plot subband boundaries of a standard DWT basis
    draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                     label_levels=label_levels)
    axes[0, level].set_title('{} level\ndecomposition'.format(level))

    # compute the 2D DWT
    c = pywt.wavedec2(x, 'db2', mode='periodization', level=level)
    # normalize each coefficient array independently for better visibility
    c[0] /= np.abs(c[0]).max()
    for detail_level in range(level):
        c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
    # show the normalized coefficients
    arr, slices = pywt.coeffs_to_array(c)
    axes[1, level].imshow(arr, cmap=plt.cm.gray)
    axes[1, level].set_title('Coefficients\n({} level)'.format(level))
    axes[1, level].set_axis_off()

plt.tight_layout()
plt.show()


"""