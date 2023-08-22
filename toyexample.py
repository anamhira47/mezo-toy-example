'''
Toy example of Mezo Algorithm to help understand intuition

'''
import numpy as np
from math import sqrt
seed_value = 42

np.random.seed(seed_value)  
"""
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
"""
def perturb_parameters(theta, epsilon, seed):
    

    # Set the random seed to ensure that we sample the same z for perturbation/update

    np.random.seed(seed)
    # go through each of the parameters and perturb them
    
    for i in range(len(theta)):
        z = np.random.normal(0, 1)
        
        theta[i] = theta[i] + epsilon * z
    return theta
# Define theta to be a lst of 50 parameters which are normalized numbers from 0,1
'''
Theta - parameters
Loss_func - loss function
Epsilon - perturbation hyperparameter
T - number of steps
B - Batch size 
lr - learning rate or lr scheduler
data- data to fit/find loss for

'''
def MeZO(theta, loss_func, T, epsilon, batch_size, lr, data, seed=None):
    losses = []
    for t in range(T):
        # Sample a batch of data
        batch = np.random.choice(data, batch_size)
        # Sample a random seed
        seed = np.random.randint(0, 100000)
        # pertrub paramerts by epsilon*N(0,1)
        theta = perturb_parameters(theta, epsilon, seed)
        # Calculate loss of perturbed parameters
        #E(N(0,1)) = 0
        # loss(theta + epsilon*N(0,1), batch)
        loss_plus = loss_func(theta, batch)
        
        #second theta is -2*epsilon*N(0,1) because we have already added epsilon*N(0,1) to theta so we need to subtract it twice
        # to get the finite difference approximation
        theta = perturb_parameters(theta, -2*epsilon, seed)
        # Calculate loss of perturbed parameters
        #E(N(0,1)) = 0
        # loss(theta - 2*epsilon*N(0,1), batch)
        loss_minus = loss_func(theta, batch)

        # Calculate the gradient approximation
        # g = (loss_plus - loss_minus) / (2*epsilon)
        # finite difference approximation
        approx_grad = (loss_plus - loss_minus) / (2*epsilon)
        # reset random number generator wtih seed s
        np.random.seed(seed)
        #change the parameters wrt to the gradient and learning rate
        for i in range(len(theta)):
            z  = np.random.normal(0, 1)
            theta[i] = theta[i] - lr * approx_grad*z
        losses.append(loss_func(theta, batch))
        
    return theta, losses


def loss_func(theta, batch):
    # Quadratic loss function
    return np.sum(theta**2)

# Initialize parameters
n = 50
theta = np.random.randn(n)

# Hyperparameters
T = 100
epsilon = 0.1
batch_size = 10
lr = 0.01
data = np.random.randn(100)

import matplotlib.pyplot as plt

# Run MeZO and store the loss at each iteration
theta, losses = MeZO(theta, loss_func, T, epsilon, batch_size, lr, data)
# Plot the loss over time
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over time using MeZO')
plt.show()
