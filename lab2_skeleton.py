from __future__ import print_function
import numpy as np

# In this first part, we just prepare our data (mnist)
# for training and testing

# import keras
from tensorflow.keras.datasets import mnist


def sigmoid(x):
    """
     This function returns the sigmoid value of input variable x
    :param x:
    :return: sigmoid result
    """
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x):
    """
     This method determines the derivative of sigmoid function
    :param x:
    :return: derivative of sigmoid
    """

    # determine sigmoid
    fx = sigmoid(x)

    return fx * (1 - fx)


# Cross Entropy for Binary class classification
def CrossEntropy(yHat, y, classes=2):
    # if binary classification
    if classes == 2:
        loss = -(y * np.log(yHat) + (1 - y) * np.log(1 - yHat))

    else:  # if multiclass classification
        loss = np.sum(y * np.log(yHat))

    return loss


def derivative_BCE(y, y_hat):
    """
    :param y: True Label ( either 0 or 1)
    :param y_hat:  Predicted value
    :return: derivative of the binary cross entropy
    """
    return (-y / y_hat) + ((1 - y) / (1 - y_hat))


########################
# TO COMPLETE
########################


class MLP:

    def __init__(self, input_units):
        # Intialize weights from random distribution of the given mu and std
        self.weights = np.random.randn(input_units, 1)
        self.bias = np.random.normal()

    def forward(self, x):

        """
        Forward pass
        :param x: input feature
        :return: Activation of the weighted sum

        """
        z_sum = np.dot(self.weights.T, x) + self.bias
        output = sigmoid(z_sum)

        return z_sum, output

    def fit(self, epochs, data, learning_rate):
        # Metric variables
        train_losses = []
        test_losses = []
        test_accuracy = []

        for epoch in range(epochs):
            # training part
            train_loss = self.train_one_epoch(data['X_train'], data['y_train'], learning_rate)

            # testing part
            test_loss, accuracy = self.test_one_epoch(data['X_test'], data['y_test'], learning_rate)

            test_accuracy.append(accuracy)
            test_losses.append(test_loss)
            train_losses.append(train_loss)
            print("Epoch {} loss: {}  Accuracy : {:.2f}%".format(epoch, train_loss, accuracy * 100))

    def train_one_epoch(self, data, targets, lr):

        loss = []
        for x, y in zip(data.T, targets.T):
            x = x.reshape([784, 1])
            y = y.reshape([1, 1])

            # get the sigmoid output and the weighted sum
            z, y_hat = self.forward(x)

            # loss
            L = CrossEntropy(y_hat, y)
            loss.append(L)
            # Da = DL W.r.t = a
            # Frist Part
            dL_da = (-y / y_hat) + ((1 - y) / (1 - y_hat))

            # Second Part
            # DZ =  DL w.r.t = z
            # dL_dz = dL_da * da_dz
            da_dz = der_sigmoid(z)
            dL_dz = dL_da * da_dz

            # Third Part
            # dL_dw = dL_dz * dz_dw
            dz_dw = x

            # Derivative Loss function with all the weights
            dL_dw = dL_dz * dz_dw

            # Derivative the Loss function w.r.t bias
            dL_db = dL_dz * 1

            # Upadte the weights now
            self.weights -= lr * dL_dw

            # update the bias
            self.bias -= lr * dL_db

        return np.array(loss).mean()

    def test_one_epoch(self, data, targets, lr):

        predicted = []
        losses = []
        counter = 500
        for x, y in zip(data.T, targets.T):
            # reshape to approprate shape
            x = x.reshape([784, 1])
            y = y.reshape([1, 1])

            # forward pass
            _, prediction = self.forward(x)

            # calculate lossss
            loss = CrossEntropy(prediction, y)
            losses.append(loss)

            # accumlate prediction
            prediction = 0 if prediction < 0.5 else 1

            # acculate the prediction
            predicted.append(prediction)

        # accuracy
        y_hat = np.array(predicted, dtype=np.float32)
        accuracy = np.mean(y_hat == targets)

        return np.array(losses).mean(), accuracy


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

    all_data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

    return all_data


if __name__ == "__main__":
    dataset = get_data()
    model = MLP(784)
    model.fit(10, dataset, 0.001)
