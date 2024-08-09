import matplotlib.pyplot as plt

from KNN import KNN
from dataset import get_dataset, get_grayscale, get_shuffled, crop_image, dropout_image
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split


num_channel = 3

kernel1 = torch.tensor([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                         [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                         [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]]])

kernel2 = torch.tensor([[[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                         [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]],
                         [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]])

kernel3 = torch.tensor([[[[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]],
                         [[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]],
                         [[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]]]])

kernel4 = torch.tensor([[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                         [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                         [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]]) / 9.0

kernel5 = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
               [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
               [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])

kernel6 = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                         [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                         [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]])

kernel7 = torch.tensor([[[[-2., -1., 0.], [-1., 1., 1.], [0., 1., 2.]],
                         [[-2., -1., 0.], [-1., 1., 1.], [0., 1., 2.]],
                         [[-2., -1., 0.], [-1., 1., 1.], [0., 1., 2.]]]])

kernel8 = torch.tensor([[[[1., 1., 1.], [1., -7., 1.], [1., 1., 1.]],
                         [[1., 1., 1.], [1., -7., 1.], [1., 1., 1.]],
                         [[1., 1., 1.], [1., -7., 1.], [1., 1., 1.]]]])

# kernel1 = torch.tensor([[[[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]],
#                          [[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]],
#                          [[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]]]])
#
# kernel2 = torch.tensor([[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
#                          [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
#                          [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]]) / 9.0

kernel1 = kernel1.reshape((1, 3, 3, 3))
kernel2 = kernel2.reshape((1, 3, 3, 3))
kernel3 = kernel3.reshape((1, 3, 3, 3))
kernel4 = kernel4.reshape((1, 3, 3, 3))

kernel5 = kernel5.reshape((1, 3, 3, 3))
kernel6 = kernel6.reshape((1, 3, 3, 3))
kernel7 = kernel7.reshape((1, 3, 3, 3))
kernel8 = kernel8.reshape((1, 3, 3, 3))

kernels = [kernel1, kernel2 , kernel3, kernel4, kernel5, kernel6, kernel7, kernel8]


def result_ker(ker, train_data_tensor):
    # ker = ker_torch[0].unsqueeze(0)
    result = F.conv2d(train_data_tensor, ker, padding=1)
    result_numpy = result.numpy()
    N = result_numpy.shape[0]
    result_numpy = result_numpy.reshape((N, 32, 32))
    result_numpy = result_numpy.reshape((N, 1024))
    print(result_numpy.shape)
    return result_numpy


if __name__ == "__main__":
    import time
    # data:
    torch.manual_seed(44)

    train_data, train_labels = get_dataset()  # N * 3 * 32 * 32, N
    train_label = np.array(train_labels)
    N = train_data.shape[0]

    if num_channel == 3:
        train_data = train_data / 255
    elif num_channel == 1:
        train_data = get_grayscale(train_data).reshape((-1, 1, 32, 32)) / 255 # N * 1 * 32 * 32

    # plt.imshow(train_data.reshape((N, 32, 32))[1], cmap='gray')
    # plt.axis('off')
    # plt.show()

    test_data, test_label = get_dataset(False)
    test_label = np.array(test_label)
    M = test_data.shape[0]

    if num_channel == 3:
        test_data = test_data / 255
    elif num_channel == 3:
        test_data = get_grayscale(test_data).reshape((-1, 1, 32, 32)) / 255

    # train_data = dropout_image(train_data)
    # test_data = dropout_image(test_data)

    # -------------------
    # train_loader and val loader
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)

    # test loader
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

    # run cnn kernal
    train_data_embedding = []
    # train_label_embedding = []

    results = result_ker(kernels[0], train_data_tensor)
    for ker_torch in kernels[1:]:
        # print(ker_torch.shape, train_data_tensor.shape)
        ######### results += result_ker(ker_torch, train_data_tensor)
        results = np.hstack((results, result_ker(ker_torch, train_data_tensor)))
        # print(result.shape)
    train_data_embedding = results
    print(train_data_embedding.shape)

    results = result_ker(kernels[0], test_data_tensor)
    for ker_torch in kernels[1:]:
        # print(ker_torch.shape, train_data_tensor.shape)
        ########## results += result_ker(ker_torch, test_data_tensor)
        results = np.hstack((results, result_ker(ker_torch, test_data_tensor)))
        # print(result.shape)
    test_data_embedding = results
    print(test_data_embedding.shape)

    print("Sdfsdsdsgsgsg")


    # for image, label in train_loader:
    #     new_train_data_embedding = model(image)
    #     new = new_train_data_embedding.detach().cpu().reshape((50, -1)).tolist()
    #     # print(len(new), len(new[0]))
    #     train_data_embedding.extend(new)
    #
    #     label = label.detach().cpu().reshape(-1).tolist()
    #     train_label_embedding.extend(label)
    #
    # train_data_embedding = np.array(train_data_embedding, dtype=np.float32)
    # train_label_embedding = np.array(train_label_embedding, dtype=np.float32)

    #-------
    # test_data_embedding = []
    # test_label_embedding = []
    # for image, label in test_loader:
    #     new_test_data_embedding = model(image)
    #     new = new_test_data_embedding.detach().cpu().reshape((50, -1)).tolist()
    #     # print(len(new), len(new[0]))
    #     test_data_embedding.extend(new)
    #
    #     label = label.detach().cpu().reshape(-1).tolist()
    #     test_label_embedding.extend(label)
    #
    # test_data_embedding = np.array(test_data_embedding, dtype=np.float32)
    # test_label_embedding = np.array(test_label_embedding, dtype=np.float32)
    # #--------

    start_time = time.time()
    accuracy = []
    k_val = range(1, 20, 2)
    for k in k_val:
        correct = 0
        total = 0
        for i in range(0, M, 100):
            prediction = KNN(k, train_data_embedding, train_label, test_data_embedding[i:i+100, :])
            corr = np.sum(prediction == test_label[i:i+100])
            correct += corr
            total += 100
            print(f"from {i} to {i + 100}, correct is {corr}")
        print(f"cor is {correct}, total is {total}")
        acc = 100 * correct / total
        accuracy.append(acc)

        curr_time = time.time()
        print(f"{curr_time-start_time} seconds, k is {k}, accuracy is {acc}%")


    print(accuracy)



    plt.plot(k_val, accuracy, marker='o', linestyle='-', color='b')
    plt.title('KNN Accuracy vs. K')
    plt.xlabel('k')
    plt.ylabel('Accuracy(%)')
    plt.grid(True)
    plt.xticks(k_val)
    plt.savefig('CNN_kernal_KNN.png')
    plt.show()
