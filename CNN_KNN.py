import matplotlib.pyplot as plt

from KNN import KNN
from dataset import get_dataset, get_grayscale
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split


num_channel = 3


class new(nn.Module):
    def __init__(self):
        super(new, self).__init__()
        # First block
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        # Second block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        # Third block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        # Linear layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),  # This 4*4 is the spatial size of the image at this stage
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

if __name__ == "__main__":
    import time
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

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

    # -------------------
    # train_loader and val loader
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(train_label, dtype=torch.long).to(device)
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)

    # train_size = int(0.8 * len(train_dataset))  # 80% for training
    # val_size = len(train_dataset) - train_size  # Remaining 20% for validation

    # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    #
    # # test loader
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(test_label, dtype=torch.long).to(device)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

# model
    model = new()
    model.eval()
    model.load_state_dict(torch.load('model_weights.pth'))
    model.to(device)

    train_data_embedding = []
    train_label_embedding = []
    for image, label in train_loader:
        new_train_data_embedding = model(image)
        new = new_train_data_embedding.detach().cpu().reshape((50, -1)).tolist()
        # print(len(new), len(new[0]))
        train_data_embedding.extend(new)

        label = label.detach().cpu().reshape(-1).tolist()
        train_label_embedding.extend(label)

    train_data_embedding = np.array(train_data_embedding, dtype=np.float32)
    train_label_embedding = np.array(train_label_embedding, dtype=np.float32)

    #-------
    test_data_embedding = []
    test_label_embedding = []
    for image, label in test_loader:
        new_test_data_embedding = model(image)
        new = new_test_data_embedding.detach().cpu().reshape((50, -1)).tolist()
        # print(len(new), len(new[0]))
        test_data_embedding.extend(new)

        label = label.detach().cpu().reshape(-1).tolist()
        test_label_embedding.extend(label)

    test_data_embedding = np.array(test_data_embedding, dtype=np.float32)
    test_label_embedding = np.array(test_label_embedding, dtype=np.float32)
    #--------

    start_time = time.time()
    accuracy = []
    k_val = range(1, 20, 2)
    for k in k_val:
        correct = 0
        total = 0
        for i in range(0, M, 100):
            prediction = KNN(k, train_data_embedding, train_label_embedding, test_data_embedding[i:i+100, :])
            corr = np.sum(prediction == test_label_embedding[i:i+100])
            correct += corr
            total += 100
            print(f"from {i} to {i + 100}, correct is {corr}")
        print(f"cor is {correct}, total is {total}")
        acc = correct / total
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
    plt.savefig('CNN_KNN.png')
    plt.show()

