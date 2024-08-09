
import numpy as np
from matplotlib import pyplot as plt, gridspec
from sklearn.model_selection import train_test_split

import dataset
from MLE import calculate_MLE_Parameters


def process_data_1():
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


def process_data_2():
    dataset.prepare_dataset()

    data, labels = dataset.get_dataset()
    data = data / 255

    test_data, label_test = dataset.get_dataset(train_split=False)
    test_data = test_data / 255

    data_grey = data.reshape(data.shape[0], -1)
    print(data_grey.shape)
    data_test_grey = test_data.reshape(test_data.shape[0], -1)

    return data_grey, data_test_grey, labels, label_test


if __name__ == '__main__':
    """fig, axes = plt.subplots(1, 10, figsize=(20, 2))  # Set up a figure with 10 subplots
    fig.suptitle('Grayscale Images', fontsize=16)  # 设置总标题
    data_grey_1, data_test_grey_1, labels_1, label_test_1 = process_data_1()
    mean_1, Sigma_1 = calculate_MLE_Parameters(data_grey_1, labels_1)
    for i in range(10):
        # Reshape each row into a 32x32 image
        img = mean_1[i].reshape(32, 32)

        # Display the image
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Class {i}')

    plt.show()"""

    data_color_1, data_test_color_1, labels_color_1, label_test_color_1 = process_data_2()
    mean_2, Sigma_2 = calculate_MLE_Parameters(data_color_1, labels_color_1)
    present_data = []
    for i in range(data_color_1.shape[0]):
        if labels_color_1[i] == len(present_data):
            present_data.append(data_color_1[i])
        if len(present_data) == 10:
            break
    
    fig = plt.figure(figsize=(20, 4))
    fig.suptitle('Color Images', fontsize=16)
    gs = gridspec.GridSpec(2, 10, height_ratios=[1, 1])  # 2行10列

    for i in range(10):
        ax1 = fig.add_subplot(gs[0, i])
        img_1 = mean_2[i].reshape(3, 32, 32).transpose(1, 2, 0)
        ax1.imshow(img_1)
        ax1.axis('off')
        ax1.set_title(f'Class {i}')

        ax2 = fig.add_subplot(gs[1, i])
        img_2 = present_data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        ax2.imshow(img_2)
        ax2.axis('off')

    plt.tight_layout()
    plt.show()

    plt.show()
