from keras.datasets import mnist

# load mnist
(train_images, train_labels),  (test_images, test_labels) = mnist.load_data()

# check dataset size
print(f"There are {train_images.shape[0]} train images and {test_images.shape[0]} test images in the MNIST dataset.")
print(f"Each image is {train_images.shape[1]} by {train_images.shape[2]} pixels encoded as an {train_images.dtype}.")

# Train an network to model mnist dataset
from keras import models
from keras import layers

# network has one 512 dense layer and one output layer with 10 outputs -  one per digit.

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# prepare input
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# prepare lables
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# train the network
network.fit(train_images, train_labels, batch_size=128, epochs=5)

(test_loss, test_accuracy) = network.evaluate(test_images, test_labels)
print(f'Model was trained with loss:[{test_loss:.4e}] and accuracy:[{test_accuracy:.4f}].')
