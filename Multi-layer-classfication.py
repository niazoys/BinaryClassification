from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# In this first part, we just prepare our data (mnist)
# for training and testing

# import keras
from tensorflow.keras.datasets import mnist


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
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # Convert to floating points ( for testing)
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

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
    print("y[{}]={}".format(i, y_train[:, i]))

    # plt.imshow(X_train[:, i].reshape(28, 28), cmap=matplotlib.cm.binary)
    # plt.axis("off")
    # plt.show()

    all_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

    return X_train, y_train, X_test, y_test


class MLP:

    def __init__(self, layers):
        self.n = None
        self.params = {}
        self.layers_dim = layers
        self.total_Layers = len(self.layers_dim)
        self.losses = []

    def intialize_parameters(self):

        # config random generator
        np.random.seed(3)

        # for each layers intialize the Weights
        for L in range(1, self.total_Layers):

            self.params["W" + str(L)] = np.random.randn(self.layers_dim[L], self.layers_dim[L - 1]) / np.sqrt(
                self.layers_dim[L - 1])
            self.params["b" + str(L)] = np.random.randn(self.layers_dim[L], 1)

    def forward(self, X):

        # calculate the forward pass results
        history = {}
        # 1. before the final layer ( all Layers upto L-1)
        A = X.T
        for idx in range(self.total_Layers - 1):
            Z = self.params["W" + str(idx + 1)].dot(A) + self.params["b" + str(idx + 1)]
            A = sigmoid(Z)

            # store computations of all layers for later use
            history["W" + str(idx + 1)] = self.params["W" + str(idx + 1)]
            history["Z" + str(idx + 1)] = Z
            history["A" + str(idx + 1)] = A

        # 2. store the final layer output
        Z = self.params["W" + str(self.total_Layers)].dot(A) + self.params["b" + str(self.total_Layers)]
        A = sigmoid(Z)
        # add tot the history
        history["W" + str(self.total_Layers)] = self.params["W" + str(self.total_Layers)]
        history["Z" + str(self.total_Layers)] = Z
        history["A" + str(self.total_Layers)] = A

        return A, history

    def backward(self, X, Y, history):

        derivatives = {}

        # add the value of the  input to history
        history["A0"] = X.T

        A = history["A" + str(self.total_Layers)]

        # Start at the last layer
        dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A)
        dZ = dA * der_sigmoid(history["Z" + str(self.total_Layers)])

        # dw is always the product of dz and previous activation
        dW = dZ.dot(history["A" + str(self.total_Layers - 1)])  # needs devison
        db = np.sum(dZ, axis=1, keepdims=True)  # needs division

        # previous derivative of dA
        dAPrev = history["W" + str(self.total_Layers)].T.dot(dZ)

        # add those derivatives to cache
        derivatives["dW" + str(self.total_Layers)] = dW
        derivatives["db" + str(self.total_Layers)] = db

        # end of las layer
        # start the other layers
        for idx in range(self.total_Layers - 1, 0, -1):
            dZ = dAPrev * der_sigmoid(history["Z" + str(idx)])
            dW = dZ.dot(history["A" + str(idx - 1)].T)
            db = np.sum(dZ, axis=1, keepdims=True)

            if idx > 1:
                # update the previous accumlated
                dAPrev = history["W" + str(idx)].T.dot(dZ)

            # store results
            derivatives["dW" + str(idx)] = dW
            derivatives["db" + str(idx)] = db
        return derivatives

    def train_model(self, X, Y, lr, epochs):

        np.random.seed(3)

        self.n = X.shape[0]

        self.layers_dim.insert(0, X.shape[1])

        self.intialize_parameters()

        # training
        for loop in range(epochs):
            A, store = self.forward(X)
            cost = np.squeeze(-(Y.dot(np.log(A.T)) + (1 - Y).dot(np.log(1 - A.T))))
            derivatives = self.backward(X, Y, store)

            for l in range(1, self.total_Layers + 1):
                self.params["W" + str(l)] = self.params["W" + str(l)] - lr * derivatives[
                    "dW" + str(l)]
                self.params["b" + str(l)] = self.params["b" + str(l)] - lr * derivatives[
                    "db" + str(l)]

            if loop % 100 == 0:
                print(cost)
                self.losses.append(cost)

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
        plt.plot(np.arange(len(self.losses)), self.losses)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()


def sigmoid(Z):
    """
     This function returns the sigmoid value of input variable x
    :param Z:
    :return: sigmoid result
    """
    return 1 / (1 + np.exp(-Z))


def der_sigmoid(x):
    """
     This method determines the derivative of sigmoid function
    :param x:
    :return: derivative of sigmoid
    """

    # determine sigmoid
    A = sigmoid(x)

    return A * (1 - A)


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = get_data()

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    print("test_y's shape: " + str(test_y.shape))
    print("train_Y's shape: " + str(train_y.shape))

    layers_dims = [46, 1]

    ann = MLP(layers_dims)

    ann.train_model(train_x.reshape(60000, 784), train_y, lr=0.1, epochs=1000)
    ann.predict(test_x.reshape(10000, 784), test_y)
    ann.plot_cost()
