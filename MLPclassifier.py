import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """A feedforward neural network with one hidden layer."""
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(input_size, hidden_size),
                                 nn.ReLU(), nn.Linear(hidden_size, output_size))
        #parameter initialization
        self.net.apply(init_xavier)

    def forward(self, X):
        return self.net(X)


def accuracy(y_hat, y):
    """Calculate the correct amounts"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Trainer(object):
    def __init__(self, net_, train_iter_, val_iter_, loss_, num_epochs_, optimizer_):
        self.net = net_
        self.train_iter = train_iter_
        self.val_iter = val_iter_
        self.loss = loss_
        self.num_epochs = num_epochs_
        self.optimizer = optimizer_

    def train_epoch(self):
        """Train the module for one epoch, return the average loss and training accuracy."""
        self.net.train()
        total_loss: float = 0
        acc: float = 0
        num: int = 0
        for X, y in self.train_iter:
            self.optimizer.zero_grad()  # clear the gradients
            y_hat = self.net(X)  # Forward Propagation
            l = self.loss(y_hat, y)  # calculate the loss
            l.mean().backward()  # Backward Propagation
            self.optimizer.step()  # Update the parameters
            total_loss += float(l.sum())
            acc += accuracy(y_hat, y)
            num += y.numel()
        return total_loss / num, acc / num

    def train(self):
        """Train the module and print the average loss int each epoch."""
        for epoch in range(self.num_epochs):
            average_loss, train_acc = self.train_epoch()
            print(f"epoch{epoch + 1}, train loss:{average_loss:f}, train accuracy:{train_acc:f}")


def show_image(batch, title, rows: int = 2, cols: int = 5, scale: int = 1.5):
    """show images in the Fashion MNIST dataset with labels as titles."""
    size = (cols * scale, rows * scale)
    fig, axes = plt.subplots(rows, cols, figsize=size)
    X, y = batch
    for i in range(rows * cols):
        row = i // cols  # calculate row index
        col = i % cols  # calcualte column index
        ax = axes[row, col]
        image = X[i].squeeze().numpy()  # shape: [1,28,28] -> [28,28]
        ax.imshow(image, cmap='gray')
        ax.set_title(title[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


class Assess(object):
    def __init__(self, net_, test_iter_, n=10):
        self.net = net_
        self.test_iter = test_iter_
        self.n = n
        self.labels = [
            't-shirt', 'trouser', 'pullover', 'dress', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot',
        ]

    def predict(self):
        """Show the true labels and the predicted labels."""
        for X, y in self.test_iter:
            break
        self.net.eval()
        with torch.no_grad():  # prohibit calculating the gradient
            labels = [self.labels[int(i)] for i in y]
            predictions = [self.labels[int(i)] for i in self.net(X).argmax(axis=1)]
            titles = ['Label:' + label + '\n' + 'Predict:' + pred for label, pred in zip(labels, predictions)]
            show_image((X, y), title=titles[0:self.n])


# definition of hyperparameters
batch_size, epochs, learning_rate, num_hidden = 128, 15, 0.1, 256

if __name__ == "__main__":
    trans = transforms.Compose([transforms.ToTensor()])  # method that transforms a picture to tensor
    # Download the FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Definite the neuron network, loss function and optimizer
    net = MLP(input_size=784, hidden_size=num_hidden, output_size=10)
    loss = nn.CrossEntropyLoss(reduction='none')  # the softmax function is included
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  # stochastic gradient descend
    # Train the module to update parameters
    trainer = Trainer(net, train_loader, test_loader, loss, epochs, optimizer)
    trainer.train()
    # Test and evaluate after training
    assess = Assess(net, test_loader)
    assess.predict()
