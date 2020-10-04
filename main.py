import numpy as np
import matplotlib.pylab as plt

from tensorflow.keras.datasets import mnist


class ANN:
    def __init__(self, layer_dim):

        # properties
        self.layer_dim = layer_dim
        self.parameters = {}
        self.L = len(self.layer_dim)
        self.n = 0
        self.costs = []

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def initialize_parameters(self):
        np.random.seed(1)

        for l in range(1, len(self.layer_dim)):
            self.parameters["W" + str(l)] = np.random.randn(self.layer_dim[l], self.layer_dim[l - 1]) / np.sqrt(
                self.layer_dim[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layer_dim[l], 1))

    def forward(self, X):

        store = {}

        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z

        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.sigmoid(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z

        return A, store

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def backward(self, X, Y, store):

        derivatives = {}

        store["A0"] = X.T

        A = store["A" + str(self.L)]
        dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A)

        dZ = dA * self.sigmoid_derivative(store["Z" + str(self.L)])
        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):

            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives

    def fit(self, X, Y, lr=0.01, epochs=200):
        np.random.seed(1)

        self.n = X.shape[0]

        self.layer_dim.insert(0, X.shape[1])
        self.initialize_parameters()
        for loop in range(epochs):

            A, store = self.forward(X)
            cost = np.squeeze(-(Y.dot(np.log(A.T)) + (1 - Y).dot(np.log(1 - A.T))) / self.n)

            derivatives = self.backward(X, Y, store)

            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - lr * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - lr * derivatives[
                    "db" + str(l)]

            if loop % 25 == 0:
                print(cost)
                self.costs.append(cost)

    def predict(self, X, Y):

        A, cache = self.forward(X)
        n = X.shape[0]
        p = np.zeros((1, n))
        for i in range(0, A.shape[1]):
            if A[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        print("Accuracy: " + str(np.sum((p == Y) / n)))

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()


def get_data():
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # total Number of pixels
    num_pixels = X_train.shape[1] * X_train.shape[2]

    # reshape input features
    X_train = X_train.reshape(X_train.shape[0], num_pixels).T
    X_test = X_test.reshape(X_test.shape[0], num_pixels).T

    # reshaping labels ( classes)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    # convert to floating points ( for training)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Convert to floating points ( for testing)
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # Normalize features
    X_train = X_train / 255
    X_test = X_test / 255

    # We want to have a binary classification: digit 0 is classified 1 and
    # all the other digits are classified 0

    # For seek of binary classification
    y_new = np.zeros(y_train.shape)
    y_new[np.where(y_train == 0.0)[0]] = 1
    y_train = y_new

    # For seek of binary classification
    y_new = np.zeros(y_test.shape)
    y_new[np.where(y_test == 0.0)[0]] = 1
    y_test = y_new

    y_train = y_train.T
    y_test = y_test.T

    #  Number of training examples
    m = X_train.shape[1]  # number of examples

    # Now, we shuffle the training set
    np.random.seed(138)
    shuffle_index = np.random.permutation(m)
    X_train, y_train = X_train[:, shuffle_index], y_train[:, shuffle_index]

    # Display one image and corresponding label
    import matplotlib
    import matplotlib.pyplot as plt

    i = 4
    print('y[{}]={}'.format(i, y_train[:, i]))

    # plt.imshow(X_train[:, i].reshape(28, 28), cmap=matplotlib.cm.binary)
    # plt.axis("off")
    # plt.show()
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = get_data()
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    layers_dims = [64, 1]
    ann = ANN(layers_dims)
    print(layers_dims)

    ann.fit(train_x.reshape(60000, 784), train_y, lr=0.001, epochs=200)
    ann.predict(test_x.reshape(10000, 784), test_y)
    ann.plot_cost()
