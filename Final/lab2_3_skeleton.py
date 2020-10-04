from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


# one-hot encode labels
def onHot(train_x, train_y, test_x, test_y):
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


def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


def compute_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1. / m) * L_sum

    return L


def feed_forward(X, params):
    cache = {}

    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid(cache["Z1"])
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

    return cache


def change_prediction_to_onehot(predictions):
    pred = np.argmax(predictions, axis=0)
    rows = np.arange(pred.size)

    one_hot = np.zeros(predictions.shape)
    one_hot[pred, rows] = 1

    return one_hot


def back_propagate(X, Y, params, cache, m_batch):
    dZ2 = cache["A2"] - Y
    dW2 = (1. / m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


def plot_training_curve(epoch, training_loss, validation_loss):
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

    cache = feed_forward(X_test, params)
    one_hot_pred = change_prediction_to_onehot(cache['A2'])
    accuracy = np.mean(one_hot_pred == Y_test)

    return accuracy


def train_model():

    for i in range(epochs):

        for j in range(batches):
            begin = j * batch_size
            end = min(begin + batch_size, X_train.shape[1] - 1)

            X = X_train[:, begin:end]
            Y = Y_train[:, begin:end]
            m_batch = end - begin

            cache = feed_forward(X, params)
            grads = back_propagate(X, Y, params, cache, m_batch)

            params["W1"] = params["W1"] - lr * grads["dW1"]
            params["b1"] = params["b1"] - lr * grads["db1"]
            params["W2"] = params["W2"] - lr * grads["dW2"]
            params["b2"] = params["b2"] - lr * grads["db2"]

        cache = feed_forward(X_train, params)
        train_cost = compute_loss(Y_train, cache["A2"])
        training_loss.append(train_cost)

        cache = feed_forward(X_val, params)
        valid_loss = compute_loss(Y_val, cache["A2"])
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
    lr = 4
    batch_size = 128
    batches = (m // batch_size)
    epochs = 100

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
