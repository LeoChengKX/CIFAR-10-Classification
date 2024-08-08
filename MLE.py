import dataset
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import multivariate_normal

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
    :return: u, theta, where u is 10 * 1024, and theta is 10 * 1024 * 1024 matrix
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


def calculate_mle_parameters(X, y):
    # Convert y to a NumPy array if it isn't already one
    y = np.array(y)
    num_classes = 10
    num_features = X.shape[1]
    mu = np.zeros((num_classes, num_features))
    sigma_matrices = np.zeros((num_classes, num_features, num_features))

    for c in range(num_classes):
        indices = np.where(y == c)[0]
        X_c = X[indices]  # Data for class c
        N_c = len(indices)  # Number of samples in class c

        if N_c > 0:
            mu[c] = np.mean(X_c, axis=0)
            variances = np.var(X_c, axis=0, ddof=0)  # Calculate variances for each feature in class c
            sigma_matrices[c] = np.diag(variances)  # Create a diagonal matrix with these variances
        else:
            mu[c] = np.nan  # Handle classes with no examples
            sigma_matrices[c] = np.nan  # Handle classes with no examples

    return mu, sigma_matrices


def prior(y: list) -> list:
    N = len(y)  # Total number of samples
    y = np.array(y)
    prior_probabilities = []

    for c in range(10):
        N_c = np.sum(y == c)  # Number of samples in class c
        prior_probabilities.append(N_c / N)

    return prior_probabilities


def predict(X: np.ndarray, mu, sigma) -> list:
    num_samples = X.shape[0]
    num_classes = 10
    log_probs = np.zeros((num_samples, num_classes))

    for c in range(num_classes):
        # Compute the log probability density function for class c
        rv = multivariate_normal(mean=mu[c], cov=sigma[c])
        log_pdf = rv.logpdf(X)

        # Add the log prior to the log likelihood
        log_probs[:, c] = log_pdf

    # Choose the class with the highest log probability
    predictions = np.argmax(log_probs, axis=1)

    return predictions.tolist()


def accuracy(y_true: list, y_pred: list):
    # Convert lists to numpy arrays for vectorized operations
    array1 = np.array(y_pred)
    array2 = np.array(y_true)

    # Ensure both arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Both lists must be of the same length.")

    # Calculate the number of correct predictions
    correct_predictions = np.sum(array1 == array2)

    # Calculate accuracy as the proportion of correct predictions
    accuracy_score = correct_predictions / len(array1)

    return accuracy_score


def one_hot_encode(labels: list, num_labels):
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def calculate_MAP_pre(X: np.ndarray, mu, sigma, priors: list) -> list:
    num_samples = X.shape[0]
    num_classes = len(priors)
    log_probs = np.zeros((num_samples, num_classes))

    for c in range(num_classes):
        # Compute the log probability density function for class c
        rv = multivariate_normal(mean=mu[c], cov=sigma[c])
        log_pdf = rv.logpdf(X)

        # Add the log prior to the log likelihood
        log_probs[:, c] = log_pdf + np.log(priors[c])

    # Choose the class with the highest log probability
    predictions = np.argmax(log_probs, axis=1)

    return predictions.tolist()


if __name__ == '__main__':
    data_grey, data_test_grey, labels, label_test = process_data()
    mu_1, sigma_1 = calculate_mle_parameters(data_grey, labels)
    mu_2, sigma_2 = calculate_MLE_Parameters(data_grey, labels)
    print(len(label_test))
    prior_prob = prior(labels)
    print(mu_1.shape)
    print(sigma_1.shape)
    prediction_map_1 = calculate_MAP_pre(data_test_grey, mu_1, sigma_1, prior_prob)
    print(prediction_map_1)
    prediction_map_2 = calculate_MAP_pre(data_test_grey, mu_2, sigma_2, prior_prob)
    print(accuracy(prediction_map_1, label_test))
    print(accuracy(prediction_map_2, label_test))
    prediction_1 = predict(data_test_grey, mu_1, sigma_1)
    prediction_2 = predict(data_test_grey, mu_2, sigma_2)
    print(accuracy(prediction_1, label_test))
    print(accuracy(prediction_2, label_test))
