def naive_relu(x):
    # A simple implementation of relu function for a 2D tensor.
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_add(x, y):
    # A simple implementation of add function for a 2D tensor.
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            x[i, j] = x[i, j] + y[i, j]
    return x


def naive_add_matrix_and_vector(x, y):
    # A simple implementation of add function for a 2D tensor and 1D vector to test broadcasting.
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = x[i, j] + y[j]
    return x


def naive_vector_dot(x, y):
    # A naive implementation of dot product for two vectors.
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


def naive_matrix_vector_dot(x, y):
    # A naive implementation of dot product between a matrix and a vector.
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


def naive_matrix_dot(x, y):
    # A naive implementation of dot product between a matrix and a vector.
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z

# Test naive relu
import numpy as np
a = np.array([[1,  2, 3,  4],
              [-1, 3, -5, 6]])
b = np.array([[4,  2, 5,  4],
              [2, 8,  -1, 6]])
print(f"c = naive_relu(a)\na = {a}\nc = {naive_relu(a)}\n")
print(f"c = naive_add(a, b)\na = {a}\nb = {b}\nc = {naive_add(a, b)}\n")

# Test numpy add and relu
print(f"c = np.maximum(a, 0)\na = {a}\nc = {np.maximum(a, 0)}\n")
print(f"c = a + b\na = {a}\nb = {b}\nc = {a + b}\n")

# Test naive_add_matrix_and_vector
b = np.array([4,  2, 5,  4])
print(f"c = naive_add_matrix_and_vector(a, b)\na = {a}\nb = {b}\nc = {naive_add_matrix_and_vector(a, b)}\n")

# Broadcasting works if tensors are of shape (a,...,n,...,m) and (n,...,m) in which case dimensions a through n are
# broadcast. Test it for numpy maximum.
a = np.random.random((64, 3, 32, 10))
b = np.random.random((32, 10))
c = np.maximum(a, b)
print(f"c = np.maximum(a, b)\na.shape = {a.shape}\nb.shape = {b.shape}\nc.shape = {c.shape}\n")

# 2.3.3
# Dot product
a = np.random.random(5)
b = np.random.random(5)
print(f"c = naive_vector_dot(a, b)\na = {a}\nb = {b}\nc = {naive_vector_dot(a, b)}\n")
print(f"c = np.dot(a, b)\na = {a}\nb = {b}\nc = {np.dot(a, b)}\n")

a = np.random.random((3, 2))
b = np.random.random(2)
print(f"c = naive_matrix_vector_dot(a, b)\na = {a}\nb = {b}\nc = {naive_matrix_vector_dot(a, b)}\n")
print(f"c = np.dot(a, b)\na = {a}\nb = {b}\nc = {np.dot(a, b)}\n")

a = np.random.random((3, 2))
b = np.random.random((2, 3))
print(f"c = naive_matrix_dot(a, b)\na = {a}\nb = {b}\nc = {naive_matrix_dot(a, b)}\n")
print(f"c = np.dot(a, b)\na = {a}\nb = {b}\nc = {np.dot(a, b)}\n")

# 2.3.4
# Reshaping
a = np.array([[0, 1],
              [2, 3],
              [4, 5]])
print(f"a = {a}\na.reshape((6,1)) = {a.reshape((6,1))}")
print(f"a = {a}\na.reshape((2,3)) = {a.reshape((2,3))}")

# Transposition
print(f"a = {a}\na.reshape((6,1)) = {a.transpose()}")


