import numpy as np

def relu(x):
    """Rectified Linear Unit activation function"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function to prevent dead neurons"""
    return np.maximum(alpha * x, x)

def softmax(x):
    """Softmax activation function"""
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
