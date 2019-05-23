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
