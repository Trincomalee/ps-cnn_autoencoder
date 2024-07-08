import torch
from torch import nn, optim
import matplotlib.pyplot as plt



def evaluate_model(model, test_loader, criterion):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    # return the loss on the test dataset
    return total_loss / len(test_loader)



def plot_training_logs(training_loss_log, test_loss_log):
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(training_loss_log, label='Training Loss', color='blue')
    ax.scatter(range(len(test_loss_log)), test_loss_log, color='red', label='Test Loss', marker='o')
    ax.set_title('Training and Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    plt.show()



def train_model(model, train_loader, test_loader, lr=1e-3, num_epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    training_loss_log = []
    test_loss_log = []

    for epoch in range(num_epochs):
        model.train()

        total_epoch_loss = 0.0
        num_batches = 0

        for tupel in train_loader:
            imgs_with_noise, imgs_without_noise = tupel

            optimizer.zero_grad()
            outputs = model(imgs_with_noise)
            loss = criterion(outputs, imgs_without_noise)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            num_batches += 1


        training_loss = total_epoch_loss / num_batches
        test_loss = evaluate_model(model, test_loader, criterion)

        training_loss_log.append(training_loss)
        test_loss_log.append(test_loss)

    plot_training_logs(training_loss_log, test_loss_log)



def visualize_results(model, test_loader):
    model.eval()
    with torch.no_grad():
        # each batch is a tuple (image_with_noise, image_without_noise)
        data = next(iter(test_loader))[0]
        outputs = model(data)

        plt.figure(figsize=(20, 4))
        for i in range(10):
            # display original images with noise from test dataset
            ax = plt.subplot(2, 10, i + 1)
            plt.imshow(data[i].view(28, 28).numpy(), cmap='gray')
            plt.axis('off')

            # display reconstruction
            ax = plt.subplot(2, 10, i + 11)
            plt.imshow(outputs[i].view(28, 28).numpy(), cmap='gray')
            plt.axis('off')
        plt.show()


