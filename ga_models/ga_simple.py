import random
from typing import Protocol, Tuple, List, Sequence
import numpy as np
from ga_models.ga_protocol import GAModel
from ga_models.activation import relu, softmax, leaky_relu

# Try to import Metal Performance Shaders (MPS) support for Apple Silicon
try:
    import torch
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if has_mps:
        print("MPS (Metal Performance Shaders) acceleration available")
except ImportError:
    has_mps = False


class SimpleModel(GAModel):
    def __init__(self, *, dims: Tuple[int, ...]):
        assert len(dims) >= 2, 'Error: dims must be two or higher.'
        self.dims = dims
        self._DNA = []
        for i, dim in enumerate(dims):
            if i < len(dims) - 1:
                # Initialize with Xavier/Glorot initialization for better convergence
                std_dev = np.sqrt(2.0 / (dim + dims[i+1]))
                self._DNA.append(np.random.normal(0, std_dev, (dim, dims[i+1])))

    def update(self, obs: Sequence) -> Tuple[int, ...]:
        x = np.array(obs, dtype=np.float32)  # Ensure float32 for better performance
        for i, layer in enumerate(self._DNA):
            # Ensure optimal dtype for computation
            layer_np = np.array(layer, dtype=np.float32)
            
            # Apply activation (not on input layer)
            x = leaky_relu(x, alpha=0.1) if i > 0 else x
            
            # Matrix multiplication
            x = x @ layer_np
        return softmax(x)

    def action(self, obs: Sequence):
        return self.update(obs).argmax()

    def mutate(self, mutation_rate) -> None:
        for layer in self._DNA:
            # Standard mutation
            mask = np.random.random(layer.shape) < mutation_rate
            if np.sum(mask) > 0:
                layer[mask] += np.random.normal(0, 0.2, size=np.sum(mask))
                
            # Strong mutations (10% chance)
            strong_mutation_mask = np.random.random(layer.shape) < (mutation_rate * 0.1)
            if np.sum(strong_mutation_mask) > 0:
                layer[strong_mutation_mask] += np.random.normal(0, 0.5, size=np.sum(strong_mutation_mask))
                
            # Rare neuron reset (1% chance) - helps escape local minima
            reset_mask = np.random.random(layer.shape) < (mutation_rate * 0.01)
            if np.sum(reset_mask) > 0:
                # Reset to new random weights using Xavier initialization
                std_dev = np.sqrt(2.0 / (layer.shape[0] + layer.shape[1]))
                layer[reset_mask] = np.random.normal(0, std_dev, size=np.sum(reset_mask))

    def __add__(self, other):
        baby_DNA = []
        # Use BLX-alpha crossover instead of simple weighted average
        # This allows exploration beyond the parents' values
        alpha = 0.3  # BLX-alpha parameter
        for mom, dad in zip(self._DNA, other._DNA):
            # Calculate min and max for each weight
            mins = np.minimum(mom, dad)
            maxs = np.maximum(mom, dad)
            range_w = maxs - mins
            
            # Extend the range by alpha in both directions
            extended_mins = mins - alpha * range_w
            extended_maxs = maxs + alpha * range_w
            
            # Generate random values within the extended range
            baby_layer = np.random.uniform(extended_mins, extended_maxs, mom.shape)
            baby_DNA.append(baby_layer.copy())
            
        baby = type(self)(dims=self.dims)
        baby.DNA = baby_DNA
        return baby

    @property
    def DNA(self):
        return self._DNA

    @DNA.setter
    def DNA(self, value):
        self._DNA = value