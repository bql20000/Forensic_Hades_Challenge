import numpy as np
import matplotlib.image as mpimg
import os
import cv2
import featureExtractor
from scipy.io import savemat

normalized_size = (384, 256)        # image size for normalization (width x height)
block_size = 32                     # size of one patch (32x32)
n_patches_each = 10000                 # number of positive / negative patches, this number result from the threshold when selecting positive patches
n_authentic_trained = 7000                  # number of authentic images
n_tampered_trained = 4000

# select number of negative patches equal to number of positive patches
def select_negative_patches():
    au_folder = 'Casia2/au/images'
    count = 0
    negative_patches = []
    for filename in os.listdir(au_folder):
        count += 1
        print(count)
        if (count > n_authentic_trained): continue
        img = cv2.imread(au_folder + '/' + filename)
        n_pick = n_patches_each / n_authentic_trained + (count <= n_patches_each % n_authentic_trained)
        for i in range(int(n_pick)):
            x = np.random.randint(img.shape[0] - block_size)
            y = np.random.randint(img.shape[1] - block_size)
            negative_patches.append(img[x: x+block_size, y:y+block_size])
            if len(negative_patches) >= n_patches_each: return np.asarray(negative_patches)
    return np.asarray(negative_patches)


def convert_to_dot_png_path(path):
    return path[:path.find('.')] + '.png'


def show(img):
    cv2.imshow('a', img)
    cv2.waitKey(0)

# check if 2 regions of a positive patch edge can be clearly recognized
# requirement: 0.25 <= Ratio of total tampered area <= 0.75
def is_good_positive_patch(edge):
    total = block_size * block_size
    tampered_area = np.sum(edge)
    ratio = tampered_area / total
    return 0.30 <= ratio <= 0.70

def select_positive_patches():
    positive_patches = []
    edges_folder = 'Casia2/tp/edges'
    image_folder = 'Casia2/tp/images'
    count = 0
    for image_path in os.listdir(image_folder):
        count += 1
        print(count)
        if (count > n_tampered_trained): continue
        # read tampered image and corresponding edge
        edge_path = convert_to_dot_png_path(image_path)
        edge = mpimg.imread(edges_folder + '/' + edge_path)
        img = cv2.imread(image_folder + '/' + image_path)

        # resize both image and edge to (256, 384)
        #img = cv2.resize(img, normalized_size)
        #edge = cv2.resize(edge, normalized_size)

        # select patches on the tampering boundaries
        stride = 16
        for x in range(0, img.shape[0], stride):
            for y in range(0, img.shape[1], stride):
                # (x,y): top left corner of the path
                if x + block_size >= img.shape[0] or y + block_size >= img.shape[1]: continue
                cur_edge = edge[x: x+block_size, y: y+block_size]
                if is_good_positive_patch(cur_edge):
                    cur_patch = img[x: x+block_size, y: y+block_size]
                    positive_patches.append(cur_patch)
                    if len(positive_patches) >= n_patches_each: return np.asarray(positive_patches)
    return np.asarray(positive_patches)


# ----------------------------------------------------------------------------------------------------------------------
# MAIN:

# select patches
pos_patches = select_positive_patches()
neg_patches = select_negative_patches()

# convert RBG patches to YCrCB channel
pos_patches = featureExtractor.convert_YCrCb(pos_patches)
neg_patches = featureExtractor.convert_YCrCb(neg_patches)

# Drop Y channel
pos_patches = pos_patches[:, :, :, 1:3]
neg_patches = neg_patches[:, :, :, 1:3]

# Daubechies Wavelet transform
pos_feature_vector = []
neg_feature_vector = []
for patch in pos_patches:
    pos_feature_vector.append(featureExtractor.daubechies_wavelet_transform(patch))
for patch in neg_patches:
    neg_feature_vector.append(featureExtractor.daubechies_wavelet_transform(patch))
pos_feature_vector = np.asarray(pos_feature_vector)
neg_feature_vector = np.asarray(neg_feature_vector)


# Save pre-processed data for training
X_train = np.concatenate((pos_feature_vector, neg_feature_vector))
y_train = np.concatenate((np.ones(n_patches_each), np.zeros(n_patches_each)))
print(X_train.shape, y_train.shape)
train_dict = {'feature vector': X_train, 'label': y_train}
savemat('preprocessed_data/train.csv', train_dict)

"""
pos_label = np.ones(pos_feature_vector.shape[0])
pos_X_train, pos_X_test, pos_y_train, pos_y_test = train_test_split(pos_feature_vector, pos_label, test_size=0.1, random_state=10)

neg_label = np.zeros(neg_feature_vector.shape[0])
neg_X_train, neg_X_test, neg_y_train, neg_y_test = train_test_split(neg_feature_vector, neg_label, test_size=0.1, random_state=10)

X_train = np.concatenate((pos_X_train, neg_X_train))
X_test = np.concatenate((pos_X_test, neg_X_test))
y_train = np.concatenate((pos_y_train, neg_y_train))
y_test = np.concatenate((pos_y_test, neg_y_test))

train_dict = {'feature vector': X_train, 'label': y_train}
test_dict = {'feature vector': X_test, 'label': y_test}

savemat('preprocessed_data/train.csv', train_dict)
savemat('preprocessed_data/test.csv', test_dict)


"""





