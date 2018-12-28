import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression

from keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
import warnings
#warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

import seaborn as sns;
#load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Datasets size")
print("Train data:", x_train.shape)
print("Test data:", x_test.shape)

print("Samples from training data:")
for i in range(0,10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i], cmap=plt.get_cmap("gray"))
    plt.title(y_train[i]);
    plt.axis('off');
    plt.show()

images_train =  []
for image_train in x_train:
    images_train.append(image_train.flatten())

images_test = []

for image_test in x_test:
    images_test.append(image_test.flatten())

images_train = np.array(images_train)
images_test = np.array(images_test)

from sklearn.neural_network import MLPClassifier

neural_network = MLPClassifier(hidden_layer_sizes=(200,100,50),random_state=1)

#neural_network.fit(images_train, y_train)
neural_network.fit(preprocessing.StandardScaler().fit_transform(images_train), y_train)

conf_matrix_neural_network = confusion_matrix(y_test, \
                                              neural_network.predict(preprocessing.StandardScaler().fit_transform(images_test)))

print("Confusion_matrix:")
print(conf_matrix_neural_network)

sns.heatmap(conf_matrix_neural_network)

acc = accuracy_score(y_test, neural_network.predict(preprocessing.StandardScaler().fit_transform(images_test)))
print("Neural network model accuracy is {0:0.2f}".format(acc))

plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.plot(neural_network.loss_curve_)
plt.title('Neural network cost function loss')

plt.xlabel('epoch'); plt.ylabel('error value'); plt.grid();
plt.show()

print("Number of connection between input and first hidden layer:")
print(np.size(neural_network.coefs_[0]))

print("Number of connection between first and second hidden layer:")
print(np.size(neural_network.coefs_[1]))

plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.imshow(np.transpose(neural_network.coefs_[0]), cmap=plt.get_cmap("gray"), aspect="auto")
plt.ylabel('neurons in first hidden layer'); plt.xlabel('input weights to neural network');

plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.imshow(np.transpose(neural_network.coefs_[1]), cmap=plt.get_cmap("gray"), aspect="auto")
plt.ylabel('neurons in second hidden layer'); plt.xlabel('neurons in first hidden layer');

plt.rcParams['figure.figsize'] = [10, 60]
m=200
for i in range(0,m):
    plt.subplot(m/2, 20, i+1)
    plt.axis('off')
    hidden_2 = np.transpose(neural_network.coefs_[0])[i]
    plt.imshow(np.reshape(hidden_2, (28,28)), cmap=plt.get_cmap("gray"),  aspect=1)

plt.show()