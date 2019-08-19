import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# ---------------------------------------------------------------------------#
# Training settings and define your hyper parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 164)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# ---------------------------------------------------------------------------#
"""
Step 1. Creating dataloader and define your data transformations

Reference:
ToTensor(): https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
Normalize(): https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Normalize
"""
train_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor.
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize a tensor image with mean and standard deviation
])

test_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor.
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize a tensor image with mean and standard deviation
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=True, download=True, transform=train_transform),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=False, transform=test_transform),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2)
# ---------------------------------------------------------------------------#
"""
Step 2. Define your network architecture(we implement LeNet here)
torch.nn.functional has a lot of useful functions(like relu, pooling, etc.)

Reference:
lecture: ConvolutionalNetworks.pdf, page 40.(LeNet)
paper: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
functional document: https://pytorch.org/docs/stable/nn.functional.html
"""


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)  # flatten
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


model = LeNet()
if args.cuda:
    device = torch.device('cuda')
    model.to(device)

# define optimizer/loss function
Loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


# learning rate scheduling
def adjust_learning_rate(optimizer, epoch):
    if epoch < 10:
        lr = 0.01
    elif epoch < 15:
        lr = 0.001
    else:
        lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ---------------------------------------------------------------------------#
"""
Step 3. Training and testing
For more details, please refer to the "pytorch-warm-up"
"""


# training function
def train(epoch):
    model.train()

    # Iterate over loader to form mini batches
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # Reset gradients
        output = model(data)
        loss = Loss(output, target)
        loss.backward()  # Compute gradient of loss with respect to all model weights
        optimizer.step()  # Update each model parameters after computing gradients
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))


# Testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        with torch.no_grad():  # We don't need to compute gradient during the test phase
            output = model(data)
        test_loss += Loss(output, target).item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# run and save model
for epoch in range(1, args.epochs + 1):
    adjust_learning_rate(optimizer, epoch)  # learning rate schedule
    train(epoch)
    test()

    savefilename = 'LeNet.tar'
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, savefilename)
