import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import matplotlib.pyplot as plt
import jax.tree_util

# Define the neural network model
def neural_network(params, inputs):
    W1, b1, W2, b2, W3, b3 = params
    hidden1 = jnp.dot(inputs, W1) + b1
    hidden1 = jnp.maximum(hidden1, 0)  # ReLU activation
    hidden2 = jnp.dot(hidden1, W2) + b2
    hidden2 = jnp.maximum(hidden2, 0)  # ReLU activation
    outputs = jnp.dot(hidden2, W3) + b3
    return outputs

# Initialize the model parameters
def initialize_params(input_dim, hidden_dim, output_dim):
    key = jax.random.PRNGKey(0)
    W1 = jax.random.normal(key, (input_dim, hidden_dim))
    b1 = jax.random.normal(key, (hidden_dim,))
    W2 = jax.random.normal(key, (hidden_dim, hidden_dim))
    b2 = jax.random.normal(key, (hidden_dim,))
    W3 = jax.random.normal(key, (hidden_dim, output_dim))
    b3 = jax.random.normal(key, (output_dim,))
    params = (W1, b1, W2, b2, W3, b3)
    return params

# Generate the input data
def generate_data(num_samples):
    key = jax.random.PRNGKey(0)
    inputs = jnp.linspace(-5, 5, num_samples)
    targets = jnp.sin(inputs)
    # targets = jnp.cos(inputs) + jnp.sin(3*inputs) - jnp.log(inputs**2)
    # targets = inputs*5 - inputs**2
    return inputs[:, None], targets[:, None]

# Define the squared loss function
def squared_loss(params, inputs, targets):
    predictions = neural_network(params, inputs)
    loss = jnp.mean((predictions - targets) ** 2)
    return loss

# Train the model
def train_model(params, inputs, targets, learning_rate, num_epochs):
    # Define the gradient function
    grad_fn = jax.grad(squared_loss, argnums=0)
    
    for epoch in range(num_epochs):
        loss = squared_loss(params, inputs, targets)
        params_grad = grad_fn(params, inputs, targets)

        # Update the parameters using gradient descent
        params = jax.tree_util.tree_map(
            lambda param, grad: param - learning_rate * grad,
            params,
            params_grad
        )
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return params

# Set the hyperparameters
input_dim = 1
hidden_dim = 10
output_dim = 1
learning_rate = 0.03
num_epochs = 2000

# Initialize the model parameters
params = initialize_params(input_dim, hidden_dim, output_dim)

# Generate the data
inputs, targets = generate_data(100)

# Train the model
theta_star = train_model(params, inputs, targets, learning_rate, num_epochs)

# Generate predictions using the trained model
predictions = neural_network(theta_star, inputs)

# Plot the original data and predictions
plt.scatter(inputs[:, 0], targets[:, 0], label='Original data')
plt.plot(inputs, predictions, color='red', label='Predictions')
plt.xlabel('Input')
plt.ylabel('Target')
plt.legend()
plt.show()

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def loss_fn(params, inputs, targets):
    predictions = neural_network(params, inputs)
    loss = jnp.mean((predictions - targets) ** 2)
    return loss

def sample_perturbation(reference_point, magnitude, key=5):
    key = jax.random.PRNGKey(key)
    perturbations = []
    for param in reference_point:
        shape = jnp.shape(param)
        direction = jax.random.normal(key, shape=shape)
        perturbation = direction * magnitude
        perturbations.append(perturbation)
    return tuple(perturbations)


# Function to plot the loss landscape in 3D
def plot_loss_landscape(ALPHA, BETA, losses):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(ALPHA, BETA, losses, cmap='inferno')
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Loss')
    plt.subplots_adjust(right=0.5)
    plt.show()

def create_weights(alpha, beta, refs, pert1, pert2):
    return jax.tree_util.tree_map(lambda pert1, pert2, refs: refs + alpha * pert1 + beta * pert2, pert1, pert2, refs)

inputs = jnp.linspace(-5, 5, 100)[:, None]
targets = jnp.sin(inputs)

# Define the reference point as the trained model parameters
reference_point = jax.tree_util.tree_map(jnp.array, theta_star)

# Sample perturbations
perturbation1 = sample_perturbation(reference_point, magnitude=0.1, key=5)
perturbation2 = sample_perturbation(reference_point, magnitude=0.1, key=10)

# Weight combinations
alpha = jnp.linspace(-5, 5, 200)
beta = jnp.linspace(-5, 5, 200)
ALPHA, BETA = jnp.meshgrid(alpha, beta)
weight_combinations = jnp.stack([ALPHA.ravel(), BETA.ravel()]).T # (100x100, 2)
alphas = weight_combinations[:, 0]
betas = weight_combinations[:, 1]

# Evaluate losses
perturbed_params = vmap(create_weights, in_axes=(0, 0, None, None, None))(alphas, betas, reference_point, perturbation1, perturbation2)
preds = vmap(neural_network, in_axes=(0, None))(perturbed_params, inputs)
losses = np.log(((preds - targets)**2).mean(1))
# vmap(neural_network, in_axes=(None, 0))(perturbed_params, inputs)

# # Plot the loss landscape
plot_loss_landscape(ALPHA, BETA, losses.reshape(200, 200))