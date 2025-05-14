# Snake AI Optimization Log

This document records all optimizations made to the Snake AI project and their impact on performance.

## Performance Summary

Our best model achieved:
- **Training score**: 53.00
- **Validation score**: 37.50
- **Best configuration**: Population size 150, mutation rate 0.03, SimpleModel architecture
- **Network architecture**: (21, 128, 64, 32, 4)

## Optimization History

### Core Architecture Improvements

1. **Neural Network Structure**
   - Started with basic (15, 50, 50, 4) architecture
   - Expanded to (21, 64, 32, 4) for better feature representation
   - Best performing: (21, 128, 64, 32, 4) with 4 layers providing optimal depth

2. **Activation Functions**
   - Replaced basic ReLU with Leaky ReLU (alpha=0.1)
   - Added proper softmax output normalization
   - Applied layer-specific activation patterns

3. **Weight Initialization**
   - Implemented He initialization for ReLU-based networks
   - Used smaller initialization values for input layer (sqrt(1/n_out))
   - Used larger values for hidden layers (sqrt(2/n_in))

### Genetic Algorithm Enhancements

1. **Selection Mechanism**
   - Upgraded to tournament selection with size 4
   - Implemented adaptive elitism (15-30%)
   - Used fitness-proportional probabilities for selection

2. **Crossover Improvements**
   - Implemented BLX-alpha crossover (alpha=0.3)
   - Added weighted parent contribution (50-80% from better parent)
   - Added layer-specific exploration rates

3. **Mutation Strategy**
   - Created dynamic mutation rates based on training phase:
     - Early phase (0-15%): 150% of base mutation rate
     - Mid phase (15-40%): 120% of base mutation rate
     - Late phase (60-80%): 60% of base mutation rate
     - Final phase (80-100%): 40% of base mutation rate
   - Added layer-specific mutation with higher rates for deeper layers
   - Implemented three mutation types: standard, strong and reset

4. **Fitness Function**
   - Enhanced reward for food eaten with exponential scaling
   - Added distance-based rewards for approaching food
   - Implemented diminishing returns for survival time
   - Added penalties for collisions and not eating

### Training Process Optimizations

1. **Parallel Processing**
   - Implemented multiprocessing pool for parallel evaluation
   - Used optimized number of processes based on CPU cores
   - Achieved ~16x speedup on 16-core systems

2. **Early Stopping**
   - Added patience-based early stopping (35 generations)
   - Implemented delta-based improvement detection (0.005 threshold)
   - Added progress estimation and time remaining calculation

3. **Memory & Performance**
   - Used float32 data type for network operations
   - Optimized matrix operations
   - Added Metal Performance Shaders detection for Apple Silicon

4. **Continued Training**
   - Implemented continuation from best models for further optimization
   - Used higher elitism rates (20-40%) for continued training
   - Applied lower mutation rates for refinement rather than exploration
   - Used larger tournament sizes (5) for stronger selection pressure

### Neural Network Size Impact

After extensive testing, we've found that:

1. **Small networks** (21, 64, 32, 4) - ~320 parameters
   - Train quickly (faster convergence)
   - Peak performance around 25-30 score
   - Good for initial exploration and testing

2. **Medium networks** (21, 128, 64, 32, 4) - ~12,000 parameters
   - Best balance of capacity vs. trainability
   - Reached highest scores (50+)
   - Optimal for this problem

3. **Large networks** (21, 256, 128, 64, 4) - ~40,000 parameters
   - Slower to train
   - More prone to overfitting
   - No significant improvement over medium network

### Failed Experiments

1. **Advanced Residual Model**
   - Tried implementing residual connections between layers
   - Performance was significantly worse (1.40-2.40 max score vs 40+ for SimpleModel)
   - Network complexity likely too high for the task

2. **Excessively Deep Networks**
   - Tried 5+ layer networks with diminishing returns
   - Found 4-layer network (21, 128, 64, 32, 4) to be optimal

## Best Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Population Size | 150-170 | 150 performed best overall |
| Mutation Rate | 0.025-0.03 | 0.03 gave best results |
| Network Architecture | (21, 128, 64, 32, 4) | 4-layer network optimal |
| Games per Evaluation | 5 | Balance between accuracy and speed |
| Early Stopping Patience | 35 | Prevents wasted computation |
| Tournament Size | 4 | Better selection pressure than size 3 |
| Crossover Method | BLX-alpha | Better exploration than simple averaging |
| Continued Training | Yes | Starting from best model significantly improves results |

## Ideal Workflow

The most effective training workflow is:

1. Use `experiment_manager.py` to train multiple models with different configurations
   ```bash
   # Train models with advanced parameters
   python experiment_manager.py  # Select option 2, then answer 'y' to advanced training
   ```

2. Select the best model for evaluation
   ```bash
   # Evaluate all models and select the best one
   python experiment_manager.py  # Select option 3
   ```

3. Continue training from the best model for additional improvement
   ```bash
   # Continue training from the best model with refined parameters
   python continue_training.py --generations 300 --pop-size 180
   ```

4. Evaluate the enhanced model
   ```bash
   # Run 30 games to thoroughly evaluate
   python ga_snake.py --model enhanced_model.pkl --games 30
   ```

## Future Improvement Ideas

1. **Curriculum Learning**: Start with easier scenarios and progressively increase difficulty
2. **Novelty Search**: Reward novel behaviors to encourage exploration
3. **Feature Engineering**: Add more sophisticated input features for better situational awareness
4. **Ensembles**: Combine multiple models for more robust decision-making
5. **Experience Replay**: Save and learn from historically good solutions
6. **Specialized Training Phases**: Train separate models for early-game and late-game strategies
7. **Hyperparameter Optimization**: Systematically search for optimal hyperparameters
