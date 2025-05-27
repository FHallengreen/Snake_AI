# Snake AI vs Human Performance Study

A comprehensive genetic algorithm-based AI system for playing Snake, with tools for comparing AI and human performance.

## 🎯 Project Overview

This project implements a neural network-based AI that learns to play Snake using genetic algorithms, and provides tools to systematically compare AI performance against human players.

### Key Features
- **Genetic Algorithm Training**: Evolves neural networks to play Snake optimally
- **Human Performance Tracking**: Tools to collect and analyze human gameplay data
- **Statistical Comparison**: Rigorous statistical analysis of AI vs human performance
- **Visualization**: Comprehensive charts and graphs for performance analysis
- **Experiment Management**: Automated workflow for conducting complete experiments

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Snake_AI
   ```

2. **Create and activate virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Method 1: Complete Experiment Workflow (Recommended)

Use the experiment manager for a guided, complete experimental setup:

```bash
python experiment_manager.py
```

This will guide you through:
1. Setting up experiment structure
2. Training multiple AI models
3. Selecting the best model
4. Collecting human performance data
5. Running statistical comparisons

### Method 2: Individual Components

#### Train AI Models
```bash
# Quick training (faster, good for testing)
python ga_train.py --quick-test

# Full training with different network sizes
python ga_train.py --network medium

# Advanced training (best results, takes longer)
python ga_train.py
```

#### Evaluate Trained Models
```bash
# Evaluate with visualization
python ga_snake.py --model experiment_data/best_ai/enhanced_model_final.pkl --display --games 10

# Run formal test evaluation
python ga_snake.py --test --test-games 30

# Compare two models
python ga_snake.py --compare experiment_data/best_model.pkl --games 20
```

#### Collect Human Performance Data
```bash
python human_performance.py
```

#### Compare AI vs Human Performance
```bash
python compare_performance.py
```

### Method 3: Play as Human

Experience the game yourself:

```bash
python snake_game.py
```

Use arrow keys to control the snake.

## 🧬 How the AI Works

### Genetic Algorithm Components

1. **Representation**: Neural networks with configurable architectures
2. **Initialization**: Population of random neural networks
3. **Evaluation**: Fitness based on game performance (score, survival, food-seeking)
4. **Selection**: Tournament selection with elitism
5. **Crossover**: BLX-alpha crossover for neural network weights
6. **Mutation**: Multi-level mutation (fine-tuning, exploration, neuron reset)
7. **Replacement**: Generational with elite preservation

### Neural Network Architecture

- **Input Layer**: 21 features including:
  - Wall distances (4 features)
  - Food direction and distance (7 features)
  - Danger detection (4 features)
  - Current velocity (2 features)
  - Game state (4 features)

- **Hidden Layers**: Configurable (default: 128, 64, 32 neurons)
- **Output Layer**: 4 actions (up, down, left, right)
- **Activation**: Leaky ReLU (hidden), Softmax (output)

### Key Algorithmic Improvements

- **Leaky ReLU vs ReLU**: Prevents "dying neurons" problem crucial in genetic algorithms
- **Advanced Fitness Function**: Balances score, survival, and food-seeking behavior
- **Dynamic Mutation Rates**: Adaptive mutation throughout evolution
- **Elite Preservation**: Maintains best individuals across generations

## 📊 Performance Analysis

### Statistical Methods Used

- **Effect size calculations**: Measure practical significance
- **Confidence intervals**: Quantify uncertainty in estimates

### Visualization Features

- Score distribution comparisons
- Learning curves for human players
- Experience level vs performance analysis
- Steps vs score efficiency plots

## 🔧 Configuration

### Training Parameters

Key parameters can be adjusted in the training scripts:

- `pop_size`: Population size (default: 150)
- `mut_rate`: Base mutation rate (default: 0.035)
- `generations`: Maximum generations (default: 170)
- `games`: Games per fitness evaluation (default: 5)

### Game Parameters

Modify game settings in the experiment configuration:

- `grid_size`: Game board size (default: 20x20)
- `max_steps`: Maximum steps per game (default: 2000)

## 📁 Project Structure

```
Snake_AI/
├── experiment_data/          # Experiment results and data
│   ├── ai_models/           # Trained AI models
│   ├── human_data/          # Human performance data
│   ├── results/             # Analysis results
│   └── best_ai/             # Best performing models
├── ga_models/               # Neural network implementations
│   ├── activation.py        # Activation functions
│   ├── ga_simple.py         # Main neural network model
│   └── ga_protocol.py       # Model interface
├── experiment_manager.py    # Main experiment workflow
├── ga_train.py             # AI training script
├── ga_snake.py             # AI evaluation script
├── human_performance.py    # Human data collection
├── compare_performance.py  # Statistical comparison
├── snake.py                # Core game logic
├── game_controller.py      # Human game interface
├── ga_controller.py        # AI game interface
└── requirements.txt        # Python dependencies
```

## 🧪 Running Experiments

### Complete Scientific Study

1. **Setup**: `python experiment_manager.py` → Option 1
2. **Train AI**: Option 2 (choose advanced training for best results)
3. **Select Best**: Option 3
4. **Collect Human Data**: Option 4 (need ~30 games from multiple players)
5. **Analyze Results**: Option 5

### Quick Performance Check

```bash
# Train a quick model
python ga_train.py --quick-test

# Test it
python ga_snake.py --test --test-games 10
```

## 🎯 Expected Results

Based on our experiments:
- **AI Average Score**: ~42 points
- **Human Average Score**: ~9 points  
- **AI Advantage**: ~4.8x better performance on average

## 🔬 Advanced Features

### Continue Training from Existing Models

```bash
python continue_training.py --model experiment_data/best_model.pkl --generations 100
```
