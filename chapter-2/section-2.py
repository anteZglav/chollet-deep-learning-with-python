import numpy as np

# 0 dimensional numpy array
x = np.array(32)
print(f"x={x} is a {x.ndim} dimensional array")

# 1 dimensional numpy array
x = np.array([1, 2, 3, 4, 5])
print(f"x={x} is a {x.ndim} dimensional array with shape ({x.shape})")

# 2 dimensional numpy array
x = np.array([[1,  2,  3,  4,  5],
              [11, 12, 13, 14, 15],
              [21, 22, 23, 24, 25]])
print(f"x={x} \n is a {x.ndim} dimensional array with shape ({x.shape})")

# 3 dimensional numpy array
x = np.array([[[1,  2,  3,  4,  5],
               [11, 12, 13, 14, 15],
               [21, 22, 23, 24, 25]],
              [[1, 2, 3, 4, 5],
               [11, 12, 13, 14, 15],
               [21, 22, 23, 24, 25]],
              [[1,  2,  3,  4,  5],
               [11, 12, 13, 14, 15],
               [21, 22, 23, 24, 25]]])
print(f"x={x} \n is a {x.ndim} dimensional array with shape ({x.shape})")

# Check dimensions of mnist dataset.
from keras.datasets import mnist
(train_images, train_labels),  (test_images, test_labels) = mnist.load_data()

# mnist is an image dataset with shape (samples, x, y) containing written digits
print(f"Mnist dataset is a {train_images.ndim} dimensional array with shape {train_images.shape} and "
      f"{train_images.dtype} datatype.")

# Display some data samples.
digit = train_images[10]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# Slice the dataset samples from 10 to 100 (not included)
my_slice = train_images[10:100]
# Alternative methods
# my_slice = train_images[10:100, :, :]
# my_slice = train_images[10:100, 0:28, 0:28]
print(f"my_slice contains first {my_slice.shape[0]} samples and is of shape {my_slice.shape}")

# Slice the bottom right quadrant
my_slice = train_images[:, 14:, 14:]
print(f"my_slice contains the bottom right quadrant of the images and is of shape {my_slice.shape}")

# Slice the middle of the images
my_slice = train_images[:, 7:-7, 7:-7]
print(f"my_slice contains the middle of the images and is of shape {my_slice.shape}")

# slice a batch of the dataset
batch = train_images[:128]
print(f"first batch is of shape {batch.shape}")

batch = train_images[128:256]
print(f"second batch is of shape {batch.shape}")

n = 10
batch = train_images[128 * n:128 * (n + 1)]
print(f"{n}-th batch is of shape {batch.shape}")
