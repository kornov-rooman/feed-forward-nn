import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from .hyper_params import *
from .models import FeedForwardNeuralNet
from .testing import test_model
from .training import train_model


def run(*, device: 'torch.device'):
    # training data and loader
    train_dataset = \
        torchvision.datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # testing data and loader
    test_dataset = \
        torchvision.datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # nn model
    nn_model = FeedForwardNeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=LEARNING_RATE)

    # so just train it
    train_model(loader=train_loader, num_epochs=NUM_EPOCHS,
                nn_model=nn_model, criterion=criterion, optimizer=optimizer, device=device)

    # then test
    test_model(loader=test_loader, nn_model=nn_model, device=device)

    # finally save nn model
    torch.save(nn_model.state_dict(), 'model.ckpt')
