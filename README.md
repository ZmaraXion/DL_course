# Train MNIST with PyTorch
This is the pytroch example and training on mnist.

# Model
[LeNet-5][http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf]

# Learning rate adjustment
I manually change the lr during training
"""
def adjust_learning_rate(optimizer, epoch):
    if epoch < 10:
        lr = 0.01
    elif epoch < 15:
        lr = 0.001
    else:
        lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
"""

# Data augmentation
1. Convert a PIL Image or numpy.ndarray to tensor.
2. Normalize a tensor image with mean and standard deviation
