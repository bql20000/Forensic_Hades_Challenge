import numpy as np
from scipy.io import loadmat, savemat
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# load pre-processed data
data = loadmat('preprocessed_data/train.csv')
X = data['feature vector']
N = X.shape[0]
dim = X.shape[1]
y = data['label'].reshape(N)

# normalize training data and save mean and covariance for further calculation
mean = np.mean(X, axis=0)
var = np.var(X, axis=0)
for i in range(N):
    X[i] = (X[i] - mean) / var

savemat('preprocessed_data/mean_and_var_of_trainset.csv', {'mean': mean, 'var': var})

# Train ANN
model = MLPClassifier((450, 300, 200, 150, 100), activation='relu', solver='adam', max_iter=50, verbose=True, early_stopping=True)
model.fit(X, y)

# save ANN
filename = 'trained_MLP_sklearn.sav'
joblib.dump(model, filename)

"""
# OneHotEncoder for labels
ohe = OneHotEncoder()
y = ohe.fit_transform(y.reshape((N,1))).toarray()

# todo: Train ANN using keras
# Building
model = Sequential()
hidden_layers = [450, 300, 200, 150, 100]
activation = 'relu'

model.add(Dense(hidden_layers[0], input_dim=dim, activation=activation))
for i in range(1, len(hidden_layers)):
    model.add(Dense(hidden_layers[i], activation=activation))
model.add(Dense(2, activation='softmax'))
model.summary()

# Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=10, batch_size=64)

# Saving
# model.save('trained_ANN.h5')

"""










