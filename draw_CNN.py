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
        # x = self.conv2(x)
        # x = self.conv3(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(24)
    torch.backends.cudnn.deterministic = True

    num_pict = 0
    import time
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # data:
    torch.manual_seed(44)

    train_data, train_labels = get_dataset()  # N * 3 * 32 * 32, N
    train_label = np.array(train_labels)
    N = train_data.shape[0]

    train_data = train_data / 255

    # plt.imshow(train_data.reshape((N, 32, 32))[1], cmap='gray')
    # plt.axis('off')
    # plt.show()

    # train_data = get_shuffled(train_data)

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(train_label, dtype=torch.long).to(device)
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    model = new()
    model.eval()
    model.load_state_dict(torch.load('model_weights.pth'))
    model.to(device)

    for image, label in train_loader:
        train_data_after_cnn = model(image)  # 20 * 64 * 16 * 16
        new = train_data_after_cnn.detach().cpu().reshape((20, 64, 16, 16)).tolist()

        label = label.detach().cpu().reshape(-1).tolist()

        for i in [1]:
            these_image = image.detach().cpu().reshape((20, 3, 32, 32)).numpy()
            this_image = np.transpose(these_image[i], (1, 2, 0))
            plt.imshow(this_image)
            plt.axis('off')
            plt.show()

            images, lab = new[i], label[i]  # 64 * 16 * 16, int
            print(f"picture {i} has label: {lab}")
            # [3, 6, 15, 18, 19, 26, 31, 34, 46, 50, 57, 58]
            for index, j in enumerate([2, 5, 14, 17, 18, 25, 30, 33, 45, 49, 56, 57]):
                plt.subplot(3, 4, index + 1)
                plt.imshow(images[j], cmap='gray')
                plt.axis('off')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()

            break
        break
