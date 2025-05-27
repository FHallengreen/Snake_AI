import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function to prevent dead neurons.
    
    Unlike standard ReLU, this allows small negative values to pass through,
    preventing neurons from becoming permanently inactive. This is crucial
    for genetic algorithms where dead neurons cannot recover without backpropagation.
    
    Performance difference from ReLU: Significant in GA context as it maintains
    gradient flow and prevents permanent neuron death.
    """
    return np.maximum(alpha * x, x)

def softmax(x):
    """
    Softmax activation function for output layer.
    
    Converts raw neural network outputs into a probability distribution
    over actions. Essential for proper action selection in reinforcement
    learning contexts.
    
    Differs from ReLU/Leaky ReLU: Used for probability distribution rather
    than activation. Ensures all outputs sum to 1.0 and are positive.
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
