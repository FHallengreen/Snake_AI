import numpy as np
from typing import Protocol, Tuple, List, Sequence
from ga_models.ga_protocol import GAModel
from ga_models.activation import leaky_relu, softmax

class SimpleModel(GAModel):
    """
    Neural network model that can be evolved using genetic algorithms.
    This class implements the representation, mutation, and crossover components of GA.
    """
    def __init__(self, *, dims: Tuple[int, ...]):
        """Initialize the neural network with the given dimensions (GA: Representation)"""
        assert len(dims) >= 2, 'Error: dims must be two or higher.'
        self.dims = dims
        self._DNA = []
        for i, dim in enumerate(dims):
            if i < len(dims) - 1:
                # Enhanced initialization - He initialization for better deep network training
                if i == 0:  # Input layer - use smaller initialization
                    std_dev = np.sqrt(1.0 / dims[i+1])
                else:  # Hidden layers - use He initialization
                    std_dev = np.sqrt(2.0 / dims[i])
                self._DNA.append(np.random.normal(0, std_dev, (dim, dims[i+1])))

    def update(self, obs: Sequence) -> Tuple[int, ...]:
        """Forward pass through the neural network"""
        x = np.array(obs, dtype=np.float32)
        for i, layer in enumerate(self._DNA):
            layer_np = np.array(layer, dtype=np.float32)
            
            # Use leaky_relu for hidden layers to prevent dead neurons
            if i > 0:
                x = leaky_relu(x, alpha=0.1)
            
            x = x @ layer_np
            
        return softmax(x)

    def action(self, obs: Sequence):
        """Select an action based on the observation"""
        return self.update(obs).argmax()

    def mutate(self, mutation_rate) -> None:
        """
        Modify the neural network weights randomly (GA: Mutation).
        
        This method applies three types of mutations:
        1. Small random adjustments to weights
        2. Stronger mutations with layer-specific scaling
        3. Random neuron resets
        """
        for i, layer in enumerate(self._DNA):
            # Base mutation rate logic (from existing code)
            mask = np.random.random(layer.shape) < mutation_rate
            if np.sum(mask) > 0:
                layer[mask] += np.random.normal(0, 0.2, size=np.sum(mask))
                
            # Layer-specific mutation rates - earlier layers more stable, later layers more plastic
            layer_mut_scale = 1.0 + (i * 0.1)  # Increase mutation impact for deeper layers
            
            # Strong mutations with layer-specific scaling
            strong_rate = mutation_rate * 0.1 * layer_mut_scale
            strong_mutation_mask = np.random.random(layer.shape) < strong_rate
            if np.sum(strong_mutation_mask) > 0:
                layer[strong_mutation_mask] += np.random.normal(0, 0.5, size=np.sum(strong_mutation_mask))
                
            # Random neuron reset with layer-specific probability
            reset_rate = mutation_rate * 0.01 * (1.0 + i * 0.2)  # Higher reset chance for deeper layers
            reset_mask = np.random.random(layer.shape) < reset_rate
            if np.sum(reset_mask) > 0:
                if i == 0:  # Input layer
                    std_dev = np.sqrt(1.0 / layer.shape[1])
                else:  # Hidden layers
                    std_dev = np.sqrt(2.0 / layer.shape[0])
                layer[reset_mask] = np.random.normal(0, std_dev, size=np.sum(reset_mask))

    def __add__(self, other):
        """
        Combine two neural networks to create a new one (GA: Crossover).
        
        This method implements BLX-alpha crossover with adaptive scaling:
        1. Interpolates between parents with weighted contributions
        2. Adds exploration beyond the parents' values
        3. Applies different exploration factors based on layer depth
        """
        baby_DNA = []
        alpha = 0.3
        
        # Random interpolation scale for each layer
        for i, (mom, dad) in enumerate(zip(self._DNA, other._DNA)):
            # Dynamic adaptation: exploration factor increases with layer depth
            layer_alpha = alpha * (1 + i * 0.05)  # Deeper layers explore more
            
            mins = np.minimum(mom, dad)
            maxs = np.maximum(mom, dad)
            range_w = maxs - mins
            
            extended_mins = mins - layer_alpha * range_w
            extended_maxs = maxs + layer_alpha * range_w
            
            # Weighted interpolation (not just uniform randomness)
            # Give more weight to the better parent (assumed to be "mom" as first arg)
            weight_mom = np.random.uniform(0.5, 0.8)  # 50-80% weight to first parent
            weight_dad = 1.0 - weight_mom
            
            # Generate offspring with weighted contribution plus exploration
            base_weights = mom * weight_mom + dad * weight_dad
            exploration = np.random.uniform(extended_mins, extended_maxs, mom.shape) - base_weights
            exploration_factor = 0.3  # How much pure exploration to add
            
            baby_layer = base_weights + exploration * exploration_factor
            baby_DNA.append(baby_layer.copy())
            
        baby = type(self)(dims=self.dims)
        baby.DNA = baby_DNA
        return baby

    @property
    def DNA(self):
        """Access the neural network weights (GA: Representation)"""
        return self._DNA

    @DNA.setter
    def DNA(self, value):
        """Set the neural network weights (GA: Representation)"""
        self._DNA = value