from keras.datasets import imdb
import numpy as np
import textwrap

# 3.4.1
# Load IMDB dataset.
# Only 1e4 most frequent woeds are kept. They are represented as intagers.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1e4)

# Check what num_words means
print(f"No word exceeds {max([max(sequence) for sequence in train_data])}")

# Decode a random review
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Start from 3 because first 3 numbers are reserved, if word is not in dict put default '?' char.
index = np.random.randint(0, train_data.shape[0])
decoded_review = ' '.join([reverse_word_index.get(word - 3, '?') for word in
                           train_data[index]])
label = train_labels[index]
print(f"Decoded {'positive' if label else 'negative'} review:\n{textwrap.fill(decoded_review)}")


# 3.4.2
# Construct one-hot encoding by setting the indexes of used words to one for each sequence.
# Question: Could we count the number of words instead?
def vectorize_sequences(sequences, dimension=int(1e4)):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Vectorize sequences
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras.models import Sequential
from keras.layers import Dense

# Two dense hidden layers with relu activation followed by a single sigmoid output layer to generate probabilities.
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(int(1e4),)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training and validation datasets.
x_val = x_train[:int(1e4)]
partial_x_train = x_train[int(1e4):]
y_val = y_train[:int(1e4)]
partial_y_train = y_train[int(1e4):]

# Train the model for 20 epochs with 512 batch size.
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val, y_val))
# Test results.
results_test = model.evaluate(x_test, y_test)
results_train = model.evaluate(partial_x_train, partial_y_train)
results_validate = model.evaluate(x_val, y_val)

print(f"This network achieved training accuracy of {results_train[1]*100:.2f}% with loss {results_train[0]}")
print(f"This network achieved validation accuracy of {results_validate[1]*100:.2f}% with loss {results_validate[0]}")
print(f"This network achieved test accuracy of {results_test[1]*100:.2f}% with loss {results_test[0]}")

# Predict a random test data member.
index = np.random.randint(0, x_test.shape[0])
prediction = model.predict(x_test[index].reshape((-1,int(1e4))))[0][0]
print(f" Model predicted {'positive' if prediction>0.5 else 'negative'} ({prediction*100:.2f}%)for x_test[{index}] "
      f"when true sentiment was {'positive' if y_test[index] else 'negative'}.")

# Plot history.
import matplotlib.pyplot as plt

history_dict = history.history
# Plot loss.
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values,     'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b',  label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# Plot accuracy.
plt.figure()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(acc_values) + 1)

plt.plot(epochs, acc_values,     'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b',  label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
