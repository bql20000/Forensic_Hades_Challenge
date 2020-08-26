import os
import cv2
from scipy.io import savemat


n_authentic_trained = 7000
n_tampered_trained = 4000

dict = {}

ntest =500
# todo: authentic images testing
au_folder = 'Casia2/au/images'
test_data_folder = 'test_data'
count = 0
for filename in os.listdir(au_folder):
    count += 1
    if count <= n_authentic_trained: continue
    if count > n_authentic_trained + ntest: continue
    print(count)
    path = au_folder + '/' + filename
    img = cv2.imread(path)
    cv2.imwrite(test_data_folder + '/' + filename, img)
    dict[filename] = 0

# todo: tampered images testing
tp_folder = 'Casia2/tp/images'
count = 0
for filename in os.listdir(tp_folder):
    count += 1
    if count <= n_tampered_trained: continue
    if count > n_tampered_trained + ntest: continue
    print(count)
    path = tp_folder + '/' + filename
    img = cv2.imread(path)
    cv2.imwrite(test_data_folder + '/' + filename, img)
    dict[filename] = 1

savemat('filename_labels.csv', dict)



