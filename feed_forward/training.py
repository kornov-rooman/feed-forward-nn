
MESSAGE = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'


def train_model(*, num_epochs, loader, nn_model, criterion, optimizer, device):
    total_step = len(loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loader):
            # move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # forward pass
            outputs = nn_model(images)
            loss = criterion(outputs, labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(MESSAGE.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
