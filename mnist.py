import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
import numpy as np
from math import sqrt
seed_value = 42

np.random.seed(seed_value)  
# Load MNIST dataset
train_data = MNIST(root='.', train=True, download=True, transform=ToTensor())
test_data = MNIST(root='.', train=False, transform=ToTensor())

# Define a simple logistic regression model
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x.view(-1, 784))

# Initialize model and optimizer
model = LogisticRegression()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define loss function
loss_func = torch.nn.CrossEntropyLoss()

# Convert theta to PyTorch parameters
theta = [param.data for param in model.parameters()]

def perturb_parameters(theta, epsilon, seed):
    

    # Set the random seed to ensure that we sample the same z for perturbation/update

    np.random.seed(seed)
    # go through each of the parameters and perturb them
    
    for i in range(len(theta)):
        z = np.random.normal(0, 1)
        
        theta[i] = theta[i] + epsilon * z
    return theta

# Define MeZO function
def MeZO(theta, loss_func, T, epsilon, batch_size, lr, data_loader, seed=None):
    losses = []
    accuracies = []
    for t in range(T):
        # Sample a batch of data
        batch = next(iter(data_loader))
        inputs, targets = batch
        # Sample a random seed
        seed = np.random.randint(1000000000)
        # Perturb parameters by epsilon*N(0,1)
        theta = perturb_parameters(theta, epsilon, seed)
        # Calculate loss of perturbed parameters
        loss_plus = loss_func(model(inputs), targets)
        # Second theta is -2*epsilon*N(0,1) because we have already added epsilon*N(0,1) to theta so we need to subtract it twice
        theta = perturb_parameters(theta, -2*epsilon, seed)
        # Calculate loss of perturbed parameters
        loss_minus = loss_func(model(inputs), targets)
        # Calculate the gradient approximation
        approx_grad = (loss_plus - loss_minus) / (2*epsilon)
        # Reset random number generator with seed s
        np.random.seed(seed)
        # Change the parameters wrt to the gradient and learning rate
        for i in range(len(theta)):
            z  = np.random.normal(0, 1)
            theta[i] = theta[i] - lr * approx_grad * z
        # Calculate loss and accuracy
        loss = loss_func(model(inputs), targets)
        losses.append(loss.item())
        preds = model(inputs).argmax(dim=1)
        accuracy = accuracy_score(targets.numpy(), preds.numpy())
        accuracies.append(accuracy)
    return theta, losses, accuracies

# Define data loader
data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
n = 50
theta = np.random.randn(n)

# Hyperparameters
T = 10000
epsilon = 0.01
batch_size = 100
lr = 0.001
#data = np.random.randn(100)

import matplotlib.pyplot as plt
# Run MeZO and store the loss and accuracy at each iteration
theta, losses, accuracies = MeZO(theta, loss_func, T, epsilon, batch_size, lr, data_loader)

# Plot the loss and accuracy over time
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss', color=color)
ax1.plot(losses, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)  
ax2.plot(accuracies, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.show()