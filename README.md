# Dreamboat Deep Learning

## Overview

Dreamboat deep learning package is designed to help students and practitioners understand the fundamental concepts and underlying principles of deep learning. It provides a flexible and customizable framework for creating and experimenting with deep learning modules. The package is optimized for fast implementation and deployment of basic neural networks on CPU, making it a great tool for learning and prototyping.

### Key Features:
1. **Educational Focus**: Helps deep learning students understand the core principles of neural networks, optimization, and loss functions.
2. **Customizability**: Users can easily extend and create their own deep learning modules to enhance their practical skills.
3. **CPU Optimization**: Designed for fast implementation and deployment of neural networks on CPU, making it suitable for small-scale experiments and learning purposes.
4. **Core Components**:
   - Optimizers
   - Loss functions
   - Neural network layers
   - Data loaders
   - Utilities

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Core Components](#core-components)
  - [Optimizers](#optimizers)
  - [Loss Functions](#loss-functions)
  - [Neural Networks](#neural-networks)
  - [Data Loaders](#data-loaders)
  - [Utils](#utils)
- [Demos](#demos)
- [Contributing](#contributing)

---

## Installation

To install the package, you can simply clone the repository and install dependencies using `pip`.

```bash
git clone https://github.com/huangwx8/DreamboatDL.git
python3 -m pip install -e .
```

You can run demos we provided to make sure this package is properly installed.

```bash
python3 demos/simple_nn.py
```

---

## Usage

Once installed, you can use the package to build and train your own neural networks. Here is a quick example of how to build a simple neural network:

```python
from dreamboat.networks.sequential import Sequential
from dreamboat.networks.linear import Linear
from dreamboat.networks.relu import ReLU

# Create a simple neural network
model = Sequential(
    Linear(28*28,256),
    ReLU(),
    Linear(256,128),
    ReLU(),
    Linear(128,10)
)
```

---

## Core Components

### Optimizers

The package includes several commonly used optimization algorithms, such as the Adam optimizer.

```python
from dreamboat.optimizers.adam import Adam
optimizer = Adam(0.001)
```

### Loss Functions

You can easily use loss functions like Mean Squared Error (MSE) or Cross-Entropy Loss.

```python
from dreamboat.losses.mse import MSELoss
loss_fn = MSELoss()
```

### Neural Networks

You can create fully customizable neural network layers, including dense layers, convolutional layers, and more.

```python
from dreamboat.networks.linear import Linear
```

### Data Loaders

The package includes data loaders for images, text, and more, to help with dataset management and batching.

```python
from dreamboat.utils.data_loader import DataLoader
```

---

## Demos

We provide several demo scripts to showcase how the package can be used in practice:

- **simple_nn.py**: Demonstrates the creation of a simple neural network and training it on mnist dataset.
- **cnn_demo.py**: Shows how to implement and train a Convolutional Neural Network (CNN) on an image dataset.
- **rnn_demo.py**: Example of using Recurrent Neural Networks (RNNs) for sequence modeling.
- **gan_demo.py**: Showcases a simple implementation of a Generative Adversarial Network (GAN). The goal of this GAN is to generate synthetic images that resemble a mnist dataset.
- **dqn_demo.py**: This demo demonstrates the use of Deep Q-Networks (DQN) for training an agent to solve a reinforcement learning task, such as navigating an environment or playing a game.

These demos are located in the `demos/` directory.

---

## Contributing

We welcome contributions! If you'd like to contribute to the project, please fork the repository, create a feature branch, and submit a pull request. For larger changes, please open an issue to discuss before starting your work.
