# In this file, I will find subspace trained by training data, dimension of subspace is
# determined by validation set.Then, I construct decision tree with training data.
# Then, I can project test data on subspace and use decision tree for test data
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import KernelPCA
import dataset
from DecisionTree import validation
from KNN import KNN


def process_data():
    dataset.prepare_dataset()

    data, labels = dataset.get_dataset()
    data = data / 255

    test_data, label_test = dataset.get_dataset(train_split=False)
    test_data = test_data / 255

    data_grey = dataset.get_shuffled(data)
    data_grey = data_grey.reshape(data_grey.shape[0], -1)
    data_test_grey = dataset.get_shuffled(test_data)
    data_test_grey = data_test_grey.reshape(data_test_grey.shape[0], -1)

    # Split data_grey to train and validation
    data_train, data_val, label_train, label_val = train_test_split(data_grey, labels, test_size=0.2, random_state=48)
    return data_train, data_val, data_test_grey, label_train, label_val, label_test


def pca_apply(X_train, K):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply PCA
    pca = PCA(K)
    X_train_pca = pca.fit_transform(X_train_scaled)

    return X_train_pca, pca, scaler


def find_K(X_train: np.ndarray, X_labels: list, val_train: np.ndarray, y_labels: list):
    candidate_K = [25]
    depths = [1, 5, 10, 100]
    best_K = 50
    best_depth = 5
    best_acc = 0
    best_pca = None
    best_scaler = None
    for k in candidate_K:
        for depth in depths:
            print(k)
            print(depth)
            X_train_pca, pca, scaler = pca_apply(X_train, k)
            val_train_scaled = scaler.fit_transform(val_train)
            val_train_pca = pca.transform(val_train_scaled)
            result = KNN(depth, X_train_pca, np.array(X_labels), val_train_pca)
            score = accuracy(result.tolist(), y_labels)
            print(depth, score)
            if score > best_acc:
                best_acc = score
                best_K = k
                best_depth = depth
                best_pca = pca
                best_scaler = scaler
    return best_K, best_depth, best_acc, best_pca, best_scaler


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

    return round(accuracy_score, 4)


def one_hot_encode(labels: list, num_labels):
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


if __name__ == "__main__":
    data_train, data_val, data_test_grey, label_train, label_val, label_test = process_data()
    best_k, best_d, best_acc, best_pca, best_scaler = find_K(data_train, label_train, data_val, label_val)
    data_train_scaled = best_scaler.transform(data_train)
    train_data_pca = best_pca.transform(data_train_scaled)
    test_data_scaled = best_scaler.transform(data_test_grey)
    test_data_pca = best_pca.transform(test_data_scaled)
    result = KNN(best_k, train_data_pca, np.array(label_train), test_data_pca)
    print(best_d)
    print(accuracy(result.tolist(), label_test))
