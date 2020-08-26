import numpy as np
import featureExtractor
import os
import cv2
import joblib
from scipy.io import loadmat


# testing
model = joblib.load('trained_MLP_sklearn.sav')

# load mean and var for normalization
mean_and_var_dict = loadmat('preprocessed_data/mean_and_var_of_trainset.csv')
mean = mean_and_var_dict['mean']
var = mean_and_var_dict['var']


count = 0
n_authentic_trained = 7000
n_tampered_trained = 4000
normalized_size = (384, 256)
block_size = 32
stride = 16
dx = [-1, -1, -1, 0, 1, 1, 1, 0, 0]
dy = [-1, 0, 1, 1, 1, 0, -1, -1, 0]
threshold = 0.9


def exceed_limitation_of_tampered_neighbors(x, y, mat):
    count = 0
    count_pos = 0
    for i in dx:
        for j in dy:
            nx = x + i
            ny = y + j
            if nx < 0 or ny < 0 or nx >= mat.shape[0] or ny >= mat.shape[1]: continue
            count_pos += mat[nx][ny]
            count += 1

    return count_pos / count > threshold

def predict(img):
    #img = cv2.resize(img, normalized_size)
    n_row = int((img.shape[0] - block_size + stride) / stride)
    n_col = int((img.shape[1] - block_size + stride) / stride)
    predict_matrix = np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
            x = i * stride
            y = j * stride
            cur_patch = img[x: x+block_size, y: y+block_size]
            cur_patch = cv2.cvtColor(cur_patch, cv2.COLOR_BGR2YCR_CB)
            cur_patch = cur_patch[:, :, 1:3]
            cur_patch = featureExtractor.daubechies_wavelet_transform(cur_patch)
            cur_patch = (cur_patch - mean) / var
            predicted_type = model.predict(cur_patch.reshape((1, -1)))[0]
            predict_matrix[i][j] = predicted_type

    #print(predict_matrix)
    for i in range(n_row):
        for j in range(n_col):
            if exceed_limitation_of_tampered_neighbors(i, j, predict_matrix): return 'tampered'

    return 'authentic'


TP = 0; FP = 0; TN = 0; FN = 0
ntest = 20
# todo: authentic images testing
au_folder = 'Casia2/au/images'
count = 0
for filename in os.listdir(au_folder):
    count += 1
    if count <= n_authentic_trained: continue
    if count > n_authentic_trained + ntest: continue
    print(count)
    img = cv2.imread(au_folder + '/' + filename)
    if predict(img) == 'authentic':
        TN += 1
    else:
        FN += 1

# todo: tampered images testing
tp_folder = 'Casia2/tp/images'
count = 0
for filename in os.listdir(tp_folder):
    count += 1
    if count <= n_tampered_trained: continue
    if count > n_tampered_trained + ntest: continue
    print(count)
    img = cv2.imread(tp_folder + '/' + filename)
    if predict(img) == 'tampered':
        TP += 1
    else:
        FP += 1

# todo: calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F_score = (2 * precision * recall) / (recall + precision)
print("Accuracy: ", accuracy)
print("F_score: ", F_score)

print("Authentic accuracy: ", TN / (TN + FN))
print("Tampered accuracy: ", precision)


