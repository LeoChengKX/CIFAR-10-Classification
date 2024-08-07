import matplotlib.pyplot as plt
from dataset import get_dataset, get_grayscale
import numpy as np
from sklearn.model_selection import train_test_split


def distance_fun(A, B):
    # l2:
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
    dist = np.sqrt(A_norm + B_norm - 2 * A.dot(B.transpose()))
    return dist


def KNN(k:int, train_data, train_labels, test_data):
    """

    :param k:
    :param train_data:   N * 1024
    :param train_labels: N
    :param test_data:    M * 1024
    :return:
    """
    M = test_data.shape[0]

    dist = distance_fun(test_data, train_data)  # N * M
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]  # M * k

    # print("????",valid_labels.shape)
    class_counts = np.zeros((M, 10), dtype=int)
    for c in range(10):
        class_counts[:, c] = np.sum(valid_labels == c, axis=1)

    predicted_labels = np.argmax(class_counts, axis=1)

    return predicted_labels


if __name__ == "__main__":
    import time

    train_data, train_labels = get_dataset()  # N * 3 * 32 * 32, N
    train_labels = np.array(train_labels)
    N = train_data.shape[0]
    train_data = get_grayscale(train_data).reshape((N, -1))  # N * 1024

    train_data, val_data, train_labels, val_label = train_test_split(train_data, train_labels, train_size=0.8)
    num_train = train_data.shape[0]
    num_val = val_data.shape[0]

    # plt.imshow(train_data.reshape((N, 32, 32))[1], cmap='gray')
    # plt.axis('off')
    # plt.show()

    test_data, test_label = get_dataset(False)
    test_label = np.array(test_label)
    M = test_data.shape[0]
    test_data = get_grayscale(test_data).reshape((M, -1))

    start_time = time.time()
    accuracy = []
    k_val = range(1, 20, 2)
    for k in k_val:
        prediction = KNN(k, train_data, train_labels, val_data)
        acc = np.sum(prediction == val_label) / num_val
        accuracy.append(acc)

        curr_time = time.time()
        print(f"{curr_time-start_time} seconds, k is {k}, accuracy is {acc}")

    print(accuracy)

    plt.plot(k_val, accuracy, marker='o', linestyle='-', color='b')
    plt.title('KNN Accuracy vs. K')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(k_val)
    plt.savefig('ori_KNN.png')
    plt.show()




