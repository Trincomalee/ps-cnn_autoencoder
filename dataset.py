import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


def prepare_dataset(batch_size=256):
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    # generate dataset without noise
    # from [0, 255] to [0, 1]
    x_train = train_dataset.data.type(torch.FloatTensor) / 255.0
    x_test = test_dataset.data.type(torch.FloatTensor) / 255.0
    noise_rate = 0.1

    # generate dataset with noise
    x_train_noisy = x_train + noise_rate * torch.randn_like(x_train)
    x_test_noisy = x_test + noise_rate * torch.randn_like(x_test)
    x_train_noisy = torch.clamp(x_train_noisy, 0.0, 1.0)
    x_test_noisy = torch.clamp(x_test_noisy, 0.0, 1.0)

    # flatten
    x_train = x_train.view((-1, 784))
    x_test = x_test.view((-1, 784))
    x_train_noisy = x_train_noisy.view((-1, 784))
    x_test_noisy = x_test_noisy.view((-1, 784))

    # Combine dataset with and without noise
    # AS A TUPEL!!!
    # The 1. element of a tupel is an image with noise
    # The 2. element of a tupel is an image without noise
    train_loader = DataLoader(TensorDataset(x_train_noisy, x_train), batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_noisy, x_test), batch_size, shuffle=False)

    return train_loader, test_loader

