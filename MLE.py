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

    return data_grey, data_test_grey, labels, label_test


def calculate_MLE_Parameters(X: np.ndarray, y: list):
    """
    :param X: train data
    :param y: train labels
    :return: u, theta, where u is 10 * 3072, and theta is 10 * 3072 * 3072 matrix
    """
    number_features = X.shape[1]
    num_of_classes = 10
    mu = np.zeros((num_of_classes, number_features))
    sigma = np.zeros((num_of_classes, number_features, number_features))
    y = np.array(y)

    for c in range(num_of_classes):
        X_c = X[y == c]  # Filter data for class c
        N_c = X_c.shape[0]  # Number of samples in class c
        mu[c] = np.mean(X_c, axis=0)  # Mean vector for class c
        # Calculate covariance matrix for class c
        centered_X_c = X_c - mu[c]
        sigma[c] = np.dot(centered_X_c.T, centered_X_c) / N_c

    return mu, sigma


def prior(y: list) -> np.array:
    N = len(y)  # Total number of samples
    prior_probabilities = np.zeros(10)

    for c in range(10):
        N_c = np.sum(y == c)  # Number of samples in class c
        prior_probabilities[c] = N_c / N

    return prior_probabilities


def predict(X: np.ndarray, mu, sigma):
    epsilon = 1e-20
    num_classes = mu.shape[0]
    num_features = mu.shape[1]
    predictions = []

    # Precompute constants
    log_two_pi = np.log(2 * np.pi)
    det_sigma = []
    inv_sigma = []
    for c in range(num_classes):
        temp = np.abs(np.linalg.det(sigma[c]))
        if temp < epsilon:
            temp = epsilon
        det_sigma.append(temp)
        inv_sigma.append(np.linalg.inv(sigma[c]))

    for x in X:
        log_probs = []
        for c in range(num_classes):
            if np.any(np.isnan(sigma[c])):
                # Skip if covariance matrix is NaN
                log_probs.append(-np.inf)
                continue

            size = num_features
            det_sigma_c = det_sigma[c]
            inv_sigma_c = inv_sigma[c]
            diff = x - mu[c]

            # Log probability density calculation
            log_prob_density = -0.5 * (size * log_two_pi + np.log(det_sigma_c) + diff.T @ inv_sigma_c @ diff)
            log_probs.append(log_prob_density)

        # Classify to the class with the highest log probability
        predictions.append(np.argmax(log_probs))
        print(len(predictions))

    return np.array(predictions)


def accuracy(y_pred: list, y_true: list):
    return np.mean(y_true == y_pred)


def one_hot_encode(labels: list, num_labels):
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


if __name__ == '__main__':
    data_grey, data_test_grey, labels, label_test = process_data()
    mu, sigma = calculate_MLE_Parameters(data_grey, labels)
    prior_prob = prior(labels)
    print(mu.shape)
    print(sigma.shape)
    print(prior_prob.shape)
    prediction = predict(data_test_grey, mu, sigma)
    print(prediction.shape)
    print(accuracy(prediction, label_test))
