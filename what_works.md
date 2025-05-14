# Snake AI: What Works and What Doesn't

This document captures our findings about which approaches produce the best results for the Snake AI project.

## Best Performing Models

The **SimpleModel** with a neural network structure of `(21, 128, 64, 32, 4)` consistently achieves the best results:

- Top score: ~48.60 in training, ~40.50 in validation
- Best hyperparameters: Population size 170, mutation rate 0.025

## What Works Well

1. **Neural Network Architecture**
   - 4-layer network (21, 128, 64, 32, 4) with the original SimpleModel
   - Leaky ReLU activation functions with alpha=0.1
   - Softmax output layer

2. **Genetic Algorithm Parameters**
   - Population sizes between 150-170
   - Mutation rates between 0.025-0.03
   - Adaptive mutation rates starting higher and decreasing over time
   - Tournament selection with tournament size 4
   - Elitism (preserving 15-20% of best individuals)

3. **Feature Engineering**
   - 21 input features including:
     - Wall distances
     - Food position (relative and absolute)
     - Danger detection
     - Current velocity
     - Food direction indicators
     - Body length (normalized)

4. **Training Approach**
   - 5 games per fitness evaluation
   - 200 generations (or early stopping)
   - BLX-alpha crossover (alpha=0.3)
   - Multiprocessing for parallel evaluation

## What Doesn't Work Well

1. **Advanced Residual Networks**
   - Despite theoretical advantages, the AdvancedModel with residual connections performs significantly worse
   - Maximum score of only 2.40 vs 48.60 with SimpleModel
   - More complex architectures struggle to learn effective strategies

2. **Excessively Deep Networks**
   - Networks with too many layers (5+ layers) perform worse
   - Deeper architecture (21, 128, 96, 64, 32, 4) shows worse results than (21, 128, 64, 32, 4)

3. **Overly Complex Fitness Functions**
   - Adding too many reward components dilutes the primary objectives
   - Exponential scaling for food eaten provides better guidance than linear rewards

## Current Issues and Limitations

The AdvancedModel implementation may have several issues:

1. **Initialization Problems**: The current initialization might create values that are too large or too small

2. **Skip Connection Implementation**: The residual connections might interfere with the learning process

3. **Architecture Mismatch**: The problem might be too simple for a complex residual architecture to be necessary

## Recommendations

1. Continue using the SimpleModel with (21, 128, 64, 32, 4) architecture
2. Maintain population sizes in the 150-170 range
3. Use mutation rates around 0.025-0.03
4. Use early stopping to prevent wasting computation when progress plateaus
5. Consider trying curriculum learning (gradually increasing difficulty)

## Future Improvements

1. Experiment with different input feature representations
2. Try more sophisticated crossover methods
3. Implement a form of novelty search to improve exploration
4. Try using a separate evaluation set during training to prevent overfitting

Old fitness:
    # Enhanced fitness function with better reward for long survival after significant food collection
    fitness = (
        # Primary goal: eating food with exponential scaling for higher scores
        (food_eaten * 20000) * (1.0 + food_eaten * 0.05) + 
        
        # Getting closer to food - more reward for distance reduction to encourage path finding
        (max_distance_reduced * 650) + 
        (food_rewards * 60) +
        
        # Survival time with diminishing returns but bonus for longer survival with high score
        (min(total_steps, 1000) * 2) +  
        (max(0, total_steps - 1000) * 0.7) +  # Increased from 0.5 to 0.7
        (food_eaten > 20) * (total_steps * 5) +  # Big bonus for surviving long with high score
        
        # Penalties
        (collisions * -2500) -  # Increased penalty for collisions
        (total_steps_without_food * 12)  # Increased penalty for not eating