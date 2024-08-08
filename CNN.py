import matplotlib.pyplot as plt
from dataset import get_dataset, get_grayscale
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split

from torchsummary import summary


BATCH_NUM = 800

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First block
        self.conv1 = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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
        x = x.view(x.size(0), -1)  # Flatten the features into a vector
        x = self.fc(x)
        # return F.softmax(x, dim=1)
        return x

# def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=10):
#     best_val_accuracy = 0
#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         for images, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#         # Validation phase
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)
#         val_accuracy = 100 * correct / total
#         best_val_accuracy = max(best_val_accuracy, val_accuracy)
#     return best_val_accuracy


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    acc = 0
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        corr, tot = evaluate_model(model, val_loader)

        curr_acc = 100 * corr / tot
        if curr_acc > acc:
            acc = curr_acc
        else:
            if curr_acc < acc - 1:
                print("early stop")
                break
        print(f"val acc: {curr_acc}%")

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

            # print(f"output is: {outputs[0]}, prediction is {predicted[0]}, correct is {labels[0]}")
            # plt.imshow(images[0].detach().cpu().reshape((32, 32)), cmap='gray')
            # plt.show()


    # print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
    return correct, total


if __name__ == "__main__":
    import time
    torch.manual_seed(44)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    summary(CNN().to(device), (3, 32, 32), device="cuda")

    train_data, train_labels = get_dataset()  # N * 3 * 32 * 32, N
    train_label = np.array(train_labels)
    N = train_data.shape[0]

    train_data = train_data / 255
    # train_data = get_grayscale(train_data).reshape((-1, 1, 32, 32)) / 255 # N * 1 * 32 * 32

    # plt.imshow(train_data.reshape((N, 32, 32))[1], cmap='gray')
    # plt.axis('off')
    # plt.show()

    test_data, test_label = get_dataset(False)
    test_label = np.array(test_label)
    M = test_data.shape[0]

    test_data = test_data / 255
    # test_data = get_grayscale(test_data).reshape((-1, 1, 32, 32)) / 255

    # -------------------
    # train_loader and val loader
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(train_label, dtype=torch.long).to(device)
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)

    train_size = int(0.8 * len(train_dataset))  # 80% for training
    val_size = len(train_dataset) - train_size  # Remaining 20% for validation

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_NUM, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_NUM, shuffle=True)

    # test loader
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(test_label, dtype=torch.long).to(device)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_NUM, shuffle=True)

    # model
    best_model = None
    best_accuracy = 0
    best_model_index = None

    model = CNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, criterion, optimizer, num_epochs=23)

    cor, tot = evaluate_model(model, test_loader)
    print(f'Accuracy for test images: {100 * cor / tot}%')

    # Save model state dictionary
    torch.save(model.state_dict(), 'model_weights.pth')


# for index, model in enumerate(models):
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #     criterion = nn.CrossEntropyLoss()
    #     accuracy = train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=10)
    #     print(f"CNN_{index} has Validation Accuracy: {accuracy}%")
    #
    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         best_model = model
    #
    #         best_model_index = index
    #
    # print(f"Best mdoel is: CNN_{best_model_index}, Validation Accuracy: {best_accuracy}%")
