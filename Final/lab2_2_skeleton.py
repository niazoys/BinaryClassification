from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


def get_data(class_1=0):
    """
    This function 1 vs many data split into training , validation and testin
    [80, 10 , 10 ] - proportion
    :param class_1:  the digit to be positive class
    :return: split data [ training , validation , test]
    """

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


def sigmoid(Z):
    """
     This function returns the sigmoid value of input variable x
    :param Z: dot product between weights and inputs
    :return: sigmoid result
    """
    return 1 / (1 + np.exp(-Z))


def compute_loss(Y, Y_hat):
    """
    Computes the cross entropy loss
    :param Y:  True labels
    :param Y_hat:  Preditions
    :return: loss
    """
    m = Y.shape[1]
    L = -(1. / m) * (np.sum(np.multiply(np.log(Y_hat), Y)) + np.sum(np.multiply(np.log(1 - Y_hat), (1 - Y))))

    return L


def feed_forward(X, params):
    """
    Forward propagation for the network
    :param X: Input features
    :param params: set of weights and bias
    :return: weighted sum and the activation of it
    """
    Z1 = np.matmul(params["W1"], X) + params["b1"]
    A1 = sigmoid(Z1)

    store = {"Z1": Z1, "A1": A1}

    return store


def back_propagate(X, Y, params, store, m_batch):
    """
     Finds the back propagation
    :param X: Input features
    :param Y: True labels
    :param params: set of parameters
    :param store: calculated values for the weighted sum and activations
    :return: derivatives
    """
    dZ1 = store["A1"] - Y

    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1}

    return grads


def test_model(X_test, Y_test, params):
    """
    Evaluates the model for un seen data

    :param X_test: test data features
    :param Y_test:  test labels
    :param params:  set of weights and bias for the trained model
    :return:  accuracy : accuracy of the result
    """
    # forward pass
    store = feed_forward(X_test, params)

    # get the max probability and change it to label one hot at the
    # digit
    pred = change_prob_to_label(store["A1"])

    # average the accuracy
    accuracy = np.mean(pred == Y_test)

    return accuracy


def change_prob_to_label(probas):
    """
    change the probs to their respective label

    :param probas: softmax logits
    :return: labels
    """
    probas[probas < 0.5] = 0
    probas[probas >= 0.5] = 1

    return probas


def split_training_validation_set(m_v, X_train, Y_train):
    """

    :param m_v: number of sample
    :param X_train: training samples
    :param Y_train: training labels
    :return: splited data
    """
    m_v = 10000
    X_val = X_train[:, :m_v]
    Y_Val = Y_train[:, :m_v]

    X_train = X_train[:, m_v:]
    Y_train = Y_train[:, m_v:]

    return X_train, Y_train, X_val, Y_Val


def plot_training_curve(epoch, training_losses, validation_losses, class_1, n_h):
    """
   
   :param epoch:  number of epochs
   :param training_losses: accumlated training loss
   :param validation_losses: accumlated valiation loss
   :param class_1: positive class digit
   :param n_h: number of hidden layers
   :return:  graph
   """
    plt.figure()
    plt.plot(range(epochs), training_losses, label="Training Loss")
    plt.plot(range(epochs), validation_losses, label="Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.legend(loc='best')
    plt.title(" 1 H - Layer(64 Neurons) Binary (" + str(epochs) + "epochs)")
    plt.savefig(
        "training-validation-losses-{0}-neuron-binary-classification-{1}-VS- all -{2}-epoches.png".format(str(n_h),
                                                                                                          str(class_1),
                                                                                                          str(
                                                                                                              epochs)))
    plt.show()


def train_model():
    """
    Training the network

    :return: parameters , train_Loss , validation_Loss
    """
    training_losses = []
    validation_losses = []
    for i in range(epochs):

        for j in range(batches):
            begin = j * batch_size
            end = min(begin + batch_size, X_train.shape[1] - 1)

            X = X_train[:, begin:end]
            Y = Y_train[:, begin:end]
            m_batch = end - begin

            store = feed_forward(X, params)
            grads = back_propagate(X, Y, params, store, m_batch)

            params["W1"] = params["W1"] - learning_rate * grads["dW1"]
            params["b1"] = params["b1"] - learning_rate * grads["db1"]

        # check the loess
        store = feed_forward(X_train, params)
        train_loss = compute_loss(Y_train, store["A1"])
        training_losses.append(train_loss)

        store = feed_forward(X_val, params)
        val_loss = compute_loss(Y_val, store["A1"])
        validation_losses.append(val_loss)
        print("Epoch {}: training loss = {},Val loss = {}".format(i + 1, train_loss, val_loss))
    return params, training_losses, validation_losses


def intializeWeights(n_x, n_h):
    """

    :param n_x:  number of features
    :param n_h: number of hidden neurons
    :return: parameters
    """
    params = {
        "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
        "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x)
    }
    return params


if __name__ == "__main__":

    class_1 = 0
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_data(class_1)

    # hyperparameters
    n_x = X_train.shape[0]  # 784
    m = X_train.shape[1]
    n_h = 64
    learning_rate = 0.003
    batch_size = 128
    batches = -(-m // batch_size)
    epochs = 100

    # initialization
    params = intializeWeights(n_x, n_h)

    # train model
    params, training_losses, validation_losses = train_model()

    # plot training curve
    plot_training_curve(epochs, training_losses, validation_losses, class_1, n_h)

    # UnSeen Data ( Test data)
    accuracy = test_model(X_test, Y_test, params)
    print("=== Accuracy = {:.2f}%=== ".format(accuracy * 100))
    print("Done.")
