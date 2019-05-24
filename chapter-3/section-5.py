from keras.datasets import reuters
import numpy as np
import textwrap
from matplotlib import pyplot as plt

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


def plot_history(history_in):
    # Plot loss.
    loss = history_in.history['loss']
    val_loss = history_in.history['val_loss']

    epochs = range(1, len(loss) + 1)
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.subplots_adjust(hspace=0.5)
    plt.plot(epochs, loss,      'bo',   label='Training loss')
    plt.plot(epochs, val_loss,  'b',    label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot accuracy.
    acc = history_in.history['acc']
    val_acc = history_in.history['val_acc']

    plt.subplot(2, 1, 2,)
    plt.plot(epochs, acc,      'bo',   label='Training accuracy')
    plt.plot(epochs, val_acc,  'b',    label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


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

# SETUP: overfitting at 9 epochs
history = model.fit(partial_x_train, partial_y_train,
                    batch_size=512,
                    epochs=9,
                    validation_data=(x_val, y_val))

# Output results.
results_test = model.evaluate(x_test, one_hot_test_labels)
results_train = model.evaluate(partial_x_train, partial_y_train)
results_validate = model.evaluate(x_val, y_val)

print(f"This network achieved training accuracy of {results_train[1]*100:.2f}% with loss {results_train[0]}")
print(f"This network achieved validation accuracy of {results_validate[1]*100:.2f}% with loss {results_validate[0]}")
print(f"This network achieved test accuracy of {results_test[1]*100:.2f}% with loss {results_test[0]}")
# This network achieved 76.71% accuracy.

# Compare network to random baseline.
# Pretty sure this is incorrect in the book because we are supposing that all the labels are equally present.
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print(f"Random approach would produce {float(np.sum(hits_array)) / len(test_labels) * 100:.2f} % accuracy.")

# Test prediction.
predictions = model.predict(x_test)
print(f"For each sample we have {predictions.shape[1]} predictions.")
index = np.random.randint(0, predictions.shape[0])
with np.printoptions(precision=2, suppress=True):
    print(f"For sample {index} the network "
          f"{'correctly' if np.argmax(predictions[index]) == np.argmax(one_hot_test_labels[index]) else 'falsely'} "
          f"predicts these probabilities"
          f"(maximum likelihood index:{np.argmax(predictions[index])}):\n{predictions[index] * 100}")

