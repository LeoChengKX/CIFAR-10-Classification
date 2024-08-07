import matplotlib.pyplot as plt
from dataset import get_dataset, get_grayscale
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 512)  # Flatten the input
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # Output layer: 10 classes

    def forward(self, x):
        x = x.view(-1, 32 * 32)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Deactivate autograd for evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')


if __name__ == "__main__":
    import time

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_data, train_labels = get_dataset()  # N * 3 * 32 * 32, N
    train_label = np.array(train_labels)
    N = train_data.shape[0]
    train_data = get_grayscale(train_data).reshape((N, -1))  # N * 1024

    # plt.imshow(train_data.reshape((N, 32, 32))[1], cmap='gray')
    # plt.axis('off')
    # plt.show()

    test_data, test_label = get_dataset(False)
    test_label = np.array(test_label)
    M = test_data.shape[0]
    test_data = get_grayscale(test_data).reshape((M, -1))

    # -------------------
    # train_loader and val loader
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_label, dtype=torch.long)
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)

    train_size = int(0.8 * len(train_dataset))  # 80% for training
    val_size = len(train_dataset) - train_size  # Remaining 20% for validation

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # test loader
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_label, dtype=torch.long)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # model
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    evaluate_model(model, test_loader)
