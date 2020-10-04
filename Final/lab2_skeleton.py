from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


def get_data(class_1=0):
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
    class_1 = float(class_1)
    y_new = np.zeros(y_train.shape)
    y_new[np.where(y_train == class_1)[0]] = 1
    y_train = y_new

    # For seek of binary classification
    y_new = np.zeros(y_test.shape)
    y_new[np.where(y_test == class_1)[0]] = 1
    y_test = y_new

    y_train = y_train.T
    y_test = y_test.T

    #  Number of training examples
    m = X_train.shape[1]  # number of examples

    # Now, we shuffle the training set
    np.random.seed(138)
    shuffle_index = np.random.permutation(m)
    X_train, y_train = X_train[:, shuffle_index], y_train[:, shuffle_index]

    # devide to validation
    # Get validation set from training set
    X_train, Y_train, X_val, Y_Val = split_training_validation_set(10000, X_train, y_train)

    return X_train, Y_train, X_val, Y_Val, X_test, y_test


def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    L = -(1. / m) * (np.sum(np.multiply(np.log(Y_hat), Y)) + np.sum(np.multiply(np.log(1 - Y_hat), (1 - Y))))

    return L


def feed_forward(X, params):

    cache = {}
    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid(cache["Z1"])

    return cache


def back_propagate(X, Y, params, cache, m_batch):
    """
     Finds the back propagation
    :param X:
    :param Y:
    :param params:
    :param cache:
    :return:
    """
    dZ1 = cache["A1"] - Y

    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1}

    return grads


def test_model(X_test, Y_test, params):
    cache = feed_forward(X_test, params)
    pred = change_prob_to_label(cache["A1"])
    accuracy = np.mean(pred == Y_test)

    return accuracy


def change_prob_to_label(probas):
    probas[probas < 0.5] = 0
    probas[probas >= 0.5] = 1
    return probas


def split_training_validation_set(m_v, X_train, Y_train):

    X_val = X_train[:, :m_v]
    Y_Val = Y_train[:, :m_v]

    X_train = X_train[:, m_v:]
    Y_train = Y_train[:, m_v:]

    return X_train, Y_train, X_val, Y_Val


def plot_training_curve(epoch, training_losses, validation_losses, class_1, n_h):
    plt.figure()
    plt.plot(range(epochs), training_losses, label="Training Loss")
    plt.plot(range(epochs), validation_losses, label="Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.legend(loc='best')
    plt.title(" 1 Neuron Binary classifier (" + str(epochs) + "epochs)")
    plt.savefig("training-validation-losses-"+str(n_h)+"-neuron-binary-classification-" + str(class_1) + "-VS- all -" + str(epochs) + "-epoches.png")
    plt.show()


def train_model():

    training_losses = []
    validation_losses = []
    for i in range(epochs):

        for j in range(batches):
            begin = j * batch_size
            end = min(begin + batch_size, X_train.shape[1] - 1)

            X = X_train[:, begin:end]
            Y = Y_train[:, begin:end]
            m_batch = end - begin

            cache = feed_forward(X, params)
            grads = back_propagate(X, Y, params, cache, m_batch)

            params["W1"] = params["W1"] - learning_rate * grads["dW1"]
            params["b1"] = params["b1"] - learning_rate * grads["db1"]

        cache = feed_forward(X_train, params)
        train_loss = compute_loss(Y_train, cache["A1"])
        training_losses.append(train_loss)

        cache = feed_forward(X_val, params)
        val_loss = compute_loss(Y_val, cache["A1"])
        validation_losses.append(val_loss)
        print("Epoch {}: training loss = {},Val loss = {}".format(i + 1, train_loss, val_loss))
    return params, training_losses, validation_losses


def intializeWeights():
    params = {
        "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
        "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x)
    }
    return params


if __name__ == "__main__":

    # change this to other digits to change 1 vs n
    class_1 = 0
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_data(class_1)

    # hyperparameters
    n_x = X_train.shape[0]  # 784
    m = X_train.shape[1]
    n_h = 1
    learning_rate = 0.003
    batch_size = 128
    batches = -(-m // batch_size)
    epochs = 100

    # initialization
    params = intializeWeights()

    # train
    params, training_losses, validation_losses = train_model()

    # plot training loss
    plot_training_curve(epochs, training_losses, validation_losses, class_1, n_h)

    # UnSeen Data ( Test data)
    accuracy = test_model(X_test, Y_test, params)
    print("=== Accuracy = {:.2f}=== %".format(accuracy * 100))
    print("Done.")
