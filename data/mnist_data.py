from tensorflow.keras.datasets import mnist

# Load MNIST image datasets and their labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and normalize the input images
train_images = train_images.reshape(-1, 784) / 255.0
test_images = test_images.reshape(-1, 784) / 255.0
