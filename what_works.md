# Snake AI: What Works and What Doesn't

This document captures our findings about which approaches produce the best results for the Snake AI project.

## Best Performing Models

The **SimpleModel** with a neural network structure of `(21, 128, 64, 32, 4)` consistently achieves the best results:

- **Original model**:
  - Training score: 52.80
  - Validation score: 46.80 ± 8.57
  - Best configuration: Population size 150, mutation rate 0.032

- **Enhanced model** (after continued training):
  - Training score: 58.00
  - Validation score: 49.80 ± 14.16
  - Maximum score: 72.00
  - Median score: 53.00
  - Continued training parameters: Population size 180, mutation rate 0.028

## What Works Best

1. **Two-Stage Training Approach**
   - Initial training to find good base models and hyperparameters
   - Continued training from the best model with adjusted parameters
   - This approach yields ~6-10% performance improvements in average score
   - Produces ~22% improvement in maximum score capability
   - Improves median performance by ~14%

2. **Continued Training Parameters**
   - Population size 180 (increased from original training)
   - Mutation rate 0.028 (slightly lower than original training)
   - Higher elitism rates (20-40%) to preserve good solutions
   - Increased games per evaluation (7 instead of 5)
   - Adaptive mutation rates that decrease over generations

3. **Neural Network Architecture**
   - 4-layer network (21, 128, 64, 32, 4) with the original SimpleModel
   - Leaky ReLU activation functions with alpha=0.1
   - Softmax output layer

4. **Genetic Algorithm Parameters**
   - Population sizes between 150-180
   - Mutation rates between 0.025-0.032
   - Tournament selection with size 5 for continued training
   - Elitism with 15-20% for initial training, 20-40% for continued training

## Optimal Evaluation Strategy

### Games Per Evaluation Analysis

The data clearly shows that using 5 games per evaluation outperforms 10 games:

| Games Per Eval | Max Score | Avg Steps | Training Efficiency |
|----------------|-----------|-----------|---------------------|
| 5 games        | 49.40     | 1310.60   | Higher              |
| 10 games       | 32.20     | 654.50    | Lower               |

**Key reasons for this performance difference:**

1. **Beneficial Noise**: 5-game evaluations introduce moderate randomness that prevents premature convergence to suboptimal strategies
   - Allows discovery of high-risk, high-reward strategies that might be filtered out with more evaluations
   - Creates genetic diversity that 10-game evaluations might eliminate too early

2. **Computational Efficiency**: With the same time budget:
   - 5-game evaluations allow the algorithm to process more individuals per generation
   - More generations can be evaluated, giving evolution more opportunities to discover improvements

3. **Exploration-Exploitation Balance**:
   - 5 games: Better balance favoring exploration of the solution space
   - 10 games: Excessive exploitation leads to early convergence and plateauing

4. **Training Progress Pattern**: 
   - 5-game training shows continuous improvement through generation 170
   - 10-game training plateaus around generation 145 at a significantly lower performance level

This finding suggests that for Snake AI training, the benefits of increased exploration outweigh the advantages of more accurate fitness assessment.

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