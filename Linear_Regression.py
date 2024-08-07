# software linear regression
import dataset
from sklearn.model_selection import train_test_split
import numpy as np

FILE_PATH = 'cifar-10-python.tar.gz'
DATASET_PATH = 'dataset/'


def process_data():
    dataset.prepare_dataset()

    data, labels = dataset.get_dataset()
    data = data / 255

    test_data, label_test = dataset.get_dataset(train_split=False)
    test_data = test_data / 255

    data_grey = dataset.get_grayscale(data)
    data_grey = data_grey.reshape(data_grey.shape[0], -1)
    data_test_grey = dataset.get_grayscale(test_data)
    data_test_grey = data_test_grey.reshape(data_test_grey.shape[0], -1)

    # Split data_grey to train and validation
    data_train, data_val, label_train, label_val = train_test_split(data_grey, labels, test_size=0.2, random_state=42)
    return data_train, data_val, data_test_grey, label_train, label_val, label_test


def softmax(z):
    exp_normalized = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_normalized / np.sum(exp_normalized, axis=1, keepdims=True)


def gradient_descent(X, y_one_hot: np.ndarray, y_pred: np.ndarray):
    m = y_one_hot.shape[0]  # Number of examples
    error = y_pred - y_one_hot
    gradient = np.dot(X.T, error) / m
    return gradient


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray):
    # avoid overflow/underflow
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)

    # Compute the logarithm of the predictions
    log_y_pred = np.log(y_pred)

    # Compute the cross-entropy loss
    cross_entropy = -np.sum(y_true * log_y_pred, axis=1)
    return cross_entropy / y_pred.shape[0]


# Convert labels to one-hot-code
def one_hot_encode(labels: list, num_labels):
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def predict(X: np.ndarray, weights: np.ndarray):
    temp = X @ weights
    probs = softmax(temp)
    return one_hot_encode(np.argmax(probs, axis=1), 10)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(y_true == y_pred)


def train_softmax(X, y: list, learning_rate, max_iterations):
    num_labels = np.max(y) + 1  # number of labels
    np.random.seed(42)
    y_one_hot = one_hot_encode(y, num_labels)
    weights = np.random.randn(X.shape[1], num_labels)
    for iteration in range(max_iterations):
        temp = X @ weights
        y_pred = softmax(temp)
        gradient = gradient_descent(X, y_one_hot, y_pred)
        weights -= learning_rate * gradient
        if iteration % 100 == 0:
            loss = cross_entropy_loss(y_pred, y_one_hot)
            print(f"Iteration {iteration}, Loss: {loss}")

    return weights


def tuned_alpha_maxiterations(X_train, y_train: list, X_val, y_val: list):
    max_it_cand = [50, 100, 200, 500]
    step_size_cand = [0.1, 0.01, 0.5, 0.05]
    max_ind = 0
    step_size_ind = 0
    best_acc = 0
    for i in range(len(max_it_cand)):
        for j in range(len(step_size_cand)):
            max_it = max_it_cand[i]
            step_size = step_size_cand[j]

            # train
            weights = train_softmax(X_train, y_train, step_size, max_it)

            # for validation accuracy
            val_pred = predict(X_val, weights)
            val_true = one_hot_encode(y_val, 10)
            temp_acc = accuracy(val_pred, val_true)

            # update
            if temp_acc > best_acc:
                best_acc = temp_acc
                max_ind = i
                step_size_ind = j
    return max_it_cand[max_ind], step_size_cand[step_size_ind], best_acc


if __name__ == '__main__':
    data_train, data_val, data_test_grey, label_train, label_val, label_test = process_data()
    max_it, step_size, best_val_acc = tuned_alpha_maxiterations(data_train, label_train, data_val, label_val)
    print(max_it, step_size)
    # 500, 0.1
    weights = train_softmax(data_train, label_train, step_size, max_it)
    test_pred = predict(data_test_grey, weights)
    test_true = one_hot_encode(label_test, 10)
    print(test_true.shape)
    print(data_train.shape)
    print(test_pred.shape)
    print(accuracy(test_pred, test_true))
