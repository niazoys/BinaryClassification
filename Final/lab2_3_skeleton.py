from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


# one-hot encode labels
def onHot(train_x, train_y, test_x, test_y):
    """

    :param train_x:  training features
    :param train_y:  training labels
    :param test_x:  test_features
    :param test_y: test_labels
    :return: one hot representation of the labels
    """
    # one-hot encode labels
    digits = 10
    examples = train_y.shape[1]
    train_y = train_y.reshape(1, examples)
    Y_new = np.eye(digits)[train_y.astype('int32')]
    train_y = Y_new.T.reshape(digits, examples)

    test_y = test_y.reshape(1, test_y.shape[1])
    Y_new = np.eye(digits)[test_y.astype('int32')]
    test_y = Y_new.T.reshape(digits, test_y.shape[1])

    return train_x, train_y, test_x, test_y


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

    # one hot encoding

    y_train = y_train.T
    y_test = y_test.T

    #  Number of training examples
    m = X_train.shape[1]  # number of examples

    # Now, we shuffle the training set
    np.random.seed(138)
    shuffle_index = np.random.permutation(m)
    X_train, y_train = X_train[:, shuffle_index], y_train[:, shuffle_index]

    X_train, Y_train, X_test, Y_test = onHot(X_train, y_train, X_test, y_test)

    X_train, Y_train, X_val, Y_val = split_training_validation_set(10000, X_train, Y_train)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def sigmoid(Z):
    """
     This function returns the sigmoid value of input variable x
    :param Z: dot product between weights and inputs
    :return: sigmoid result
    """
    return 1 / (1 + np.exp(-Z))


def compute_loss(Y, Y_hat):
    """
     Cross entropy loss calculation
    :param Y: True Labels
    :param Y_hat: predicted labels
    :return:loss
    """
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1. / m) * L_sum

    return L


def feed_forward(X, params):

    store = {}

    store["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    store["A1"] = sigmoid(store["Z1"])
    store["Z2"] = np.matmul(params["W2"], store["A1"]) + params["b2"]
    store["A2"] = np.exp(store["Z2"]) / np.sum(np.exp(store["Z2"]), axis=0)

    return store


def change_prediction_to_onehot(predictions):

    """
    changes predictions to onehot
    :param predictions: predicted probs
    :return:  one hot
    """
    pred = np.argmax(predictions, axis=0)
    rows = np.arange(pred.size)

    one_hot = np.zeros(predictions.shape)
    one_hot[pred, rows] = 1

    return one_hot


def back_propagate(X, Y, params, store, m_batch):

    """
        Finds the back propagation
       :param X: Input features
       :param Y: True labels
       :param params: set of parameters
       :param store: calculated values for the weighted sum and activations
       :return: derivatives
   """

    dZ2 = store["A2"] - Y
    dW2 = (1. / m_batch) * np.matmul(dZ2, store["A1"].T)
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(store["Z1"]) * (1 - sigmoid(store["Z1"]))
    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


def plot_training_curve(epoch, training_loss, validation_loss):
    """
    :param epoch:  number of epochs
    :param training_losses: accumlated training loss
    :param validation_losses: accumlated valiation loss
    :return:  graph
    """
    plt.figure()
    plt.plot(range(epochs), training_loss, label="Training Loss")
    plt.plot(range(epochs), validation_loss, label="Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.legend(loc='best')
    plt.title(" 1 H - Layer(64 Neurons) multiclass (" + str(epochs) + "epochs)")
    plt.savefig("training-validation-losses-64-neuron-multiclass-classification-" + str(epochs) + "-epochs.png")
    plt.show()


def test_model(X_test, Y_test, params):
    """
    :param X_test: test_features
    :param Y_test: test_labels
    :param params: weights and bias
    :return: parameters , train_Loss , validation_Loss
    """

    store = feed_forward(X_test, params)
    one_hot_pred = change_prediction_to_onehot(store['A2'])
    accuracy = np.mean(one_hot_pred == Y_test)

    return accuracy


def train_model():
    """
       Training the network

       :return: parameters , train_Loss , validation_Loss
       """
    for i in range(epochs):

        for j in range(batches):
            begin = j * batch_size
            end = min(begin + batch_size, X_train.shape[1] - 1)

            X = X_train[:, begin:end]
            Y = Y_train[:, begin:end]
            m_batch = end - begin

            store = feed_forward(X, params)
            grads = back_propagate(X, Y, params, store, m_batch)

            params["W1"] = params["W1"] - lr * grads["dW1"]
            params["b1"] = params["b1"] - lr * grads["db1"]
            params["W2"] = params["W2"] - lr * grads["dW2"]
            params["b2"] = params["b2"] - lr * grads["db2"]

        store = feed_forward(X_train, params)
        train_cost = compute_loss(Y_train, store["A2"])
        training_loss.append(train_cost)

        store = feed_forward(X_val, params)
        valid_loss = compute_loss(Y_val, store["A2"])
        validation_loss.append(valid_loss)

        print("Epoch {}: training loss = {}, test loss = {}".format(i + 1, train_cost, valid_loss))
    return params, training_loss, validation_loss


def intializeWeights():
    params = {"W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
              "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
              "W2": np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
              "b2": np.zeros((digits, 1)) * np.sqrt(1. / n_h)}
    return params


if __name__ == "__main__":
    # get training , validation and test-data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_data()

    # hyperparameters
    n_x = X_train.shape[0]  # 784
    m = X_train.shape[1]
    n_h = 64
    digits = 10
    lr = 0.1
    batch_size = 128
    batches = (m // batch_size)
    epochs = 500

    # initialization
    params = intializeWeights()
    # train
    training_loss = []
    validation_loss = []

    # train model
    params, training_loss, validation_loss = train_model()

    # plot training curve
    plot_training_curve(epochs, training_loss, validation_loss)

    # test model
    accuracy = test_model(X_test, Y_test, params)
    print("=== Accuracy = {:.2f}==".format(accuracy * 100))
    print("Done.")
