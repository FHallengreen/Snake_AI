import random
import numpy as np
from typing import Tuple, Sequence
from ga_models.ga_protocol import GAModel
from ga_models.activation import relu, softmax, leaky_relu

class AdvancedModel(GAModel):
    def __init__(self, *, dims: Tuple[int, ...], use_residual=True):
        assert len(dims) >= 3, 'Error: dims must be three or higher for advanced model'
        self.dims = dims
        self.use_residual = use_residual
        
        # Main network weights
        self._DNA = []
        
        # Initialize with He/Kaiming initialization for ReLU networks
        for i, dim in enumerate(dims):
            if i < len(dims) - 1:
                # He initialization: std = sqrt(2/n_in)
                std_dev = np.sqrt(2.0 / dim)
                self._DNA.append(np.random.normal(0, std_dev, (dim, dims[i+1])))
        
        # Optional residual connections for deeper networks
        self._residual_DNA = []
        if use_residual and len(dims) > 3:
            for i in range(len(dims) - 3):
                # Connect layer i to layer i+2 (skip one layer)
                if dims[i] == dims[i+2]:  # Same dimensions required for residual
                    self._residual_DNA.append(np.eye(dims[i]) * 0.1)  # Identity init with small scale
                else:
                    # Projection matrix for different dimensions
                    self._residual_DNA.append(np.random.normal(0, 0.01, (dims[i], dims[i+2])))

    def update(self, obs: Sequence) -> Tuple[int, ...]:
        x = np.array(obs)
        layer_outputs = [x]  # Store intermediate activations for residual connections
        
        for i, layer in enumerate(self._DNA):
            # Apply activations (not on input layer)
            if i > 0:
                x = leaky_relu(x, alpha=0.1)
            
            # Process through current layer
            x = x @ layer
            
            # Add residual connection if applicable
            if self.use_residual and i >= 2 and (i-2) < len(self._residual_DNA):
                # Add residual connection from 2 layers back (skip connection)
                residual = layer_outputs[i-2] @ self._residual_DNA[i-2]
                
                # Make sure dimensions match for addition
                if residual.shape == x.shape:
                    x = x + residual
                    
            # Store this layer's output
            layer_outputs.append(x)
            
        return softmax(x)

    def action(self, obs: Sequence):
        return self.update(obs).argmax()

    def mutate(self, mutation_rate) -> None:
        # Mutate main network
        for layer in self._DNA:
            # Standard mutations (common)
            mask = np.random.random(layer.shape) < mutation_rate
            if np.sum(mask) > 0:
                layer[mask] += np.random.normal(0, 0.2, size=np.sum(mask))
            
            # Strong mutations (occasional)
            strong_mask = np.random.random(layer.shape) < (mutation_rate * 0.1)
            if np.sum(strong_mask) > 0:
                layer[strong_mask] += np.random.normal(0, 0.5, size=np.sum(strong_mask))
            
            # Reset mutations (rare)
            reset_mask = np.random.random(layer.shape) < (mutation_rate * 0.01)
            if np.sum(reset_mask) > 0:
                std_dev = np.sqrt(2.0 / layer.shape[0])  # He initialization
                layer[reset_mask] = np.random.normal(0, std_dev, size=np.sum(reset_mask))
                
        # Mutate residual connections
        if self.use_residual:
            for layer in self._residual_DNA:
                mask = np.random.random(layer.shape) < (mutation_rate * 0.5)  # Lower rate for residuals
                if np.sum(mask) > 0:
                    layer[mask] += np.random.normal(0, 0.1, size=np.sum(mask))

    def __add__(self, other):
        baby_DNA = []
        baby_residual_DNA = []
        
        # BLX-alpha crossover for main network
        alpha = 0.3
        for mom, dad in zip(self._DNA, other._DNA):
            mins = np.minimum(mom, dad)
            maxs = np.maximum(mom, dad)
            range_w = maxs - mins
            
            # Extend boundaries
            extended_mins = mins - alpha * range_w
            extended_maxs = maxs + alpha * range_w
            
            # Generate offspring
            baby_layer = np.random.uniform(extended_mins, extended_maxs, mom.shape)
            baby_DNA.append(baby_layer.copy())
            
        # Create new model
        baby = type(self)(dims=self.dims, use_residual=self.use_residual)
        baby.DNA = (baby_DNA, [])
        
        # Handle residual connections if enabled
        if self.use_residual and other.use_residual:
            for mom, dad in zip(self._residual_DNA, other._residual_DNA):
                # BLX-alpha crossover for residual
                mins = np.minimum(mom, dad)
                maxs = np.maximum(mom, dad)
                range_w = maxs - mins
                
                extended_mins = mins - alpha * range_w
                extended_maxs = maxs + alpha * range_w
                
                baby_res_layer = np.random.uniform(extended_mins, extended_maxs, mom.shape)
                baby_residual_DNA.append(baby_res_layer.copy())
                
            baby.DNA = (baby_DNA, baby_residual_DNA)
            
        return baby

    @property
    def DNA(self):
        if self.use_residual:
            return (self._DNA, self._residual_DNA)
        return self._DNA

    @DNA.setter
    def DNA(self, value):
        if isinstance(value, tuple) and len(value) == 2:
            self._DNA, res_dna = value
            if self.use_residual and res_dna:
                self._residual_DNA = res_dna
        else:
            self._DNA = value
