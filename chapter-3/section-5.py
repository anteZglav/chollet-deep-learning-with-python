from keras.datasets import reuters
import numpy as np
import textwrap

# 3.5.1
# Classify reuters newswires. A single-label multiclass classification problem.
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def decode_sequence(sequnce):
    # A function to decode a sequence.
    word_index = reuters.get_word_index()
    inverse_word_index = dict([(value, key) for (key,value) in word_index.items()])
    return ' '.join([inverse_word_index.get(i - 3, '?') for i in sequnce])


# Decode a random sequence.
# index = np.random.randint(0, train_data.shape[0])
# print(f"Decoded newswire[{index}]  of class {train_labels[index]}:"
#       f"\n{textwrap.fill(decode_sequence(train_data[index]))}")

def vectorize_sequences(sequences, dimension=10000):
    # Vectorizes sequences into one hot vectors of size dimension.
    results = np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(lables, dimension=46):
    results = np.zeros((len(lables), dimension))
    for i,label in enumerate(lables):
        results[i,label] = 1.
    return results


one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
# Alternative:
# from keras.utils import to_categorical
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

from keras import models
from keras import layers
# Construct model.
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Separate training data into train and validation sets.
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train,
                    batch_size=512,
                    epochs=20,
                    validation_data=(x_val,y_val))

from matplotlib import pyplot as plt

# Plot loss.
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss,      'bo',   label='Training loss')
plt.plot(epochs, val_loss,  'b',    label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy.
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.figure()
plt.plot(epochs, acc,      'bo',   label='Training accuracy')
plt.plot(epochs, val_acc,  'b',    label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
