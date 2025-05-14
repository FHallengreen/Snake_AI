# Snake Game AI with Genetic Algorithms vs Human Players

## Abstract
This study investigates the application of Genetic Algorithms (GA) to evolve neural networks capable of playing the classic Snake game, with performance compared to human players across various skill levels. The research explores whether evolutionary algorithms can produce gaming agents that outperform humans in decision-making speed, strategic planning, and overall score achievement. Using a controlled game environment with identical parameters for both AI and human players, this study quantifies performance differences through statistical analysis of game scores and survival duration. The findings contribute to our understanding of evolutionary algorithms' effectiveness in developing game-playing strategies that may exceed human capabilities in dynamic, constrained environments.

## Introduction
Video games present well-defined environments for testing artificial intelligence approaches against human cognitive abilities. The Snake game, with its simple rules yet complex strategic requirements, provides an ideal testbed for comparing evolutionary algorithms with human decision-making processes. Genetic algorithms, inspired by natural selection, represent a promising approach to developing game-playing strategies without explicit programming.

This research implements a neural network whose weights are evolved through a genetic algorithm, allowing the AI to develop strategies for maximizing score while avoiding obstacles. The comparative performance between this evolved AI and human players yields insights into the strengths and limitations of evolutionary computation in game environments where decision speed, pattern recognition, and long-term planning are all critical factors.

## Research Question/Problem Formulation
The objective of this project is to evaluate whether a Genetic Algorithm (GA) can evolve a neural network that plays the Snake game more effectively than human players. The Snake game challenges an agent to maximize its score by collecting food while avoiding obstacles. The GA-trained AI leverages evolutionary techniques to optimize its decision-making, while human players rely on their reflexes, pattern recognition, and planning abilities. The performance of both AI and humans will be measured by their average scores across multiple games, allowing for a direct comparison between the two approaches.

### Hypothesis
- **Hypothesis (H1):** The GA-trained AI achieves a significantly higher average score in the Snake game than human players.  
- **Null Hypothesis (H0):** There is no significant difference in the average scores between the GA-trained AI and human players.

### Research Sub-questions
1. Can a Genetic Algorithm train a neural network to play Snake more effectively than human players?
2. How does the performance gap between AI and humans vary with the human player's experience level?
3. What specific aspects of gameplay (avoiding collisions, efficiently collecting food, etc.) show the greatest difference between AI and human approaches?
4. Does the AI demonstrate strategies or techniques that human players don't typically utilize?

## Methods

### Experimental Setup
This study employs a quantitative experimental research design comparing AI and human performance in identical game environments. The Snake game serves as the control environment, with performance measured through objective metrics including score (food items collected) and survival duration (steps taken before game termination).

### Game Environment Implementation
The Snake game is implemented with the following controlled parameters:
- Grid size: 20x20 cells
- Snake movement: Up, Down, Left, Right (cardinal directions only)
- Game over conditions: Hitting the wall or the snake's own body
- Scoring: +1 for each food item collected

### GA Model Development
The Genetic Algorithm training process includes:
- Neural network architecture: Input layer (15 neurons), Hidden layers (50, 50 neurons), Output layer (4 neurons)
- Input features: Distance to walls, food position relative to snake, danger detection in immediate vicinity, current direction
- Population size: Varied between 50 and 100 individuals
- Mutation rates: Varied between 0.01 and 0.1
- Selection method: Roulette wheel selection with elitism (top 10% preserved)
- Crossover: Weighted averaging of parent neural networks
- Fitness function: Based on score, survival steps, and food proximity
- Training duration: 100 generations with 3 evaluation games per individual

### Human Performance Measurement Protocol
Human performance data is collected through:
- Participant recruitment with varying experience levels in gaming
- Collection of player demographics and self-reported experience level (1-10 scale)
- Controlled testing environment to minimize external variables
- Multiple play sessions per player (5 games minimum)
- Identical game parameters as used for AI training
- Recording of scores and game duration for each session
- Observation of gameplay strategies employed

### Data Analysis Methodology
Statistical comparison between AI and human performance includes:
- Two-sample t-test for score comparison to determine statistical significance
- Mann-Whitney U test for non-parametric comparison to account for potential non-normal distributions
- Analysis of score and step distributions across participant experience levels
- Visualization of performance metrics using box plots and histograms
- Correlation analysis between experience level and performance difference with AI

## Analysis
The analysis of GA-trained AI versus human performance focuses on three key dimensions:

### Performance Metrics Comparison
The primary analysis compares performance metrics between AI and human players:
- Average scores achieved (food items collected)
- Maximum scores reached in best attempts
- Average survival steps before game termination
- Score progression over time (learning curve)

For human participants, performance is further segmented by experience level to identify potential correlations between gaming experience and performance relative to the AI.

### Statistical Significance Testing
Statistical tests are employed to determine whether observed differences between AI and human performance are significant:
- Two-sample t-test examines if the average scores differ significantly
- Mann-Whitney U test provides a non-parametric alternative for comparing distributions
- ANOVA examines variance across different experimental conditions (mutation rates, population sizes)
- P-values < 0.05 are considered statistically significant

### Strategic Pattern Analysis
Beyond numerical metrics, gameplay patterns are analyzed to understand strategic differences:
- Food collection efficiency (steps per food item)
- Space utilization patterns (heatmap of movements)
- Risk-taking behavior (proximity to walls/body maintained)
- Long-term planning vs. reactive decision making

## Findings
The findings from the experimental comparison reveal several key insights:

### Performance Comparison Results
- The GA-trained AI achieved an average score of [X] compared to human average of [Y]
- Statistical analysis shows [significant/non-significant] difference (p-value = [Z])
- AI performance was most superior in [specific aspect], while humans performed better in [specific aspect]
- Experience level correlation: [description of relationship found]

### Strategic Differences
- AI demonstrated consistent patterns of [specific strategies]
- Human players showed greater adaptability in [specific situations]
- The neural network evolved unexpected strategies including [examples]

### Training Parameter Impact
- Population size of [optimal size] produced the highest-performing models
- Mutation rate of [optimal rate] balanced exploration and exploitation most effectively
- Performance plateaued after approximately [X] generations

## Conclusion
This research demonstrates that genetic algorithms can effectively evolve neural networks to play the Snake game at a level [comparative statement to human performance]. The findings support/refute the hypothesis that GA-trained AI can outperform human players in this specific game environment.

The most significant contribution is the demonstration that [key insight about evolutionary algorithms and game playing]. The performance gap between AI and humans was most pronounced in [specific aspect], suggesting that evolutionary algorithms excel at [specific strength].

Limitations of this study include [limitations], while future research could explore [potential directions]. Overall, this research advances our understanding of evolutionary algorithms' potential for developing game-playing strategies and contributes to the broader field of computational intelligence in gaming environments.

## Technical Implementation
This research was implemented using Python with the following technologies:
- Python 3.8+
- NumPy for numerical operations
- Pygame for game environment
- Pandas for data analysis
- Matplotlib and SciPy for statistical analysis and visualization

### Project Structure
- `snake.py`: Core Snake game implementation
- `game_controller.py`: Game controller interface and human controller implementation
- `ga_controller.py`: AI controller using the trained model
- `ga_models/`: Neural network implementation and GA model definitions
- `ga_train.py`: Genetic algorithm training script
- `ga_snake.py`: Script to run the trained AI model
- `human_performance.py`: Tool for tracking human player performance
- `compare_performance.py`: Statistical comparison of AI vs human performance
- `analyze.py`: Statistical analysis utilities
- `plot_results.py`: Visualization tools

### Key Components and Usage
The implementation allows for:
1. Training the AI model with variable parameters
2. Human gameplay and performance recording
3. AI gameplay using trained models
4. Statistical analysis and comparison of performance data
5. Visualization of results for analysis
