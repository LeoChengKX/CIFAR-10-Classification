import tarfile
import os.path as path
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F

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


def visualize(data, cmap='gray'):
    # plt.imshow(np.transpose(get_grayscale(data)[6], (1, 2, 0)))
    plt.imshow(data.transpose(1, 2, 0))
    plt.axis('off')
    plt.show()


def get_grayscale(image: np.ndarray):
    R = image[:, 0, :, :]
    G = image[:, 1, :, :]
    B = image[:, 2, :, :]
    gray_images = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return gray_images


def get_shuffled(image: np.ndarray):
    # bs, 3, img_size, img_size
    img_size = image.shape[2]
    per = torch.randperm(img_size ** 2)
    flat = image.reshape((image.shape[0], 3, img_size ** 2))
    flat = flat[:, :, per]
    return flat.reshape(image.shape[0], 3, img_size, img_size)


def crop_image(image: np.ndarray, size=0):
    return image[:, :, size:(32 - size), size:(32 - size)]


def dropout_image(image: np.ndarray, probability=0.1):
    mask = np.random.rand(image.shape[0], 32, 32) < probability
    mask = np.stack([mask, mask, mask], axis=1)

    black = np.zeros_like(image)
    return np.where(mask, black, image)


if __name__ == '__main__':
    prepare_dataset()

    data, labels = get_dataset()

    print(labels[0])
    print(dropout_image(data).shape)
    visualize(dropout_image(data)[1], cmap='color')
