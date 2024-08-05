import tarfile
import os.path as path
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

FILE_PATH = 'cifar-10-python.tar.gz'
DATASET_PATH = 'dataset/'


def prepare_dataset():
    if not os.path.isdir("dataset/cifar-10-batches-py"):
        with tarfile.open(FILE_PATH, 'r:gz') as tar:
            tar.extractall(path=DATASET_PATH)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def get_dataset(train_split=True):
    data = []
    labels = []
    if train_split:
        for batch in range(1, 6):
            batch_path = os.path.join(path.join(DATASET_PATH, "cifar-10-batches-py"), f'data_batch_{batch}')
            with open(batch_path, 'rb') as f:
                batch_dict = pickle.load(f, encoding='bytes')
                data.append(batch_dict[b'data'])
                labels += batch_dict[b'labels']

            # Convert data to numpy array and reshape
        data = np.vstack(data).reshape((-1, 3, 32, 32)).astype('uint8')

    else:
        batch_path = os.path.join(path.join(DATASET_PATH, "cifar-10-batches-py"), f'test_batch')
        with open(batch_path, 'rb') as f:
            batch_dict = pickle.load(f, encoding='bytes')
            data = batch_dict[b'data']
            labels = batch_dict[b'labels']

            data = np.vstack(data).reshape((-1, 3, 32, 32)).astype('float')

    return data, labels


def visualize(data):
    # plt.imshow(np.transpose(get_grayscale(data)[6], (1, 2, 0)))
    plt.imshow(get_grayscale(data)[1], cmap='gray')
    plt.axis('off')
    plt.show()


def get_grayscale(image: np.ndarray):
    R = image[:, 0, :, :]
    G = image[:, 1, :, :]
    B = image[:, 2, :, :]
    gray_images = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return gray_images


if __name__ == '__main__':
    prepare_dataset()

    data, labels = get_dataset()

    print(labels[0])
    visualize(data)
