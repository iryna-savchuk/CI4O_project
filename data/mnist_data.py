from tensorflow.keras.datasets import mnist
from keras.utils import np_utils

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and normalize the input images
train_images = train_images.reshape(-1, 784) / 255.0
test_images = test_images.reshape(-1, 784) / 255.0

# Preprocess class labels
#train_labels = np_utils.to_categorical(train_labels, 10)
#test_labels = np_utils.to_categorical(test_labels, 10)