I apologize for the oversight. It seems there was a misunderstanding in your query—you mentioned that I’m missing the `analyze_results.py` script we added, but I believe you’re pointing out that it wasn’t properly integrated into the README as requested, or that the README might have included unnecessary details beyond replication instructions. Since your README is intended solely for details on how to replicate and run the project (with results and other details reserved for the report), I’ll provide an updated version below that includes `analyze_results.py` and focuses strictly on replication and running instructions.

Here’s the revised README for your "Snake AI vs Human Performance Study" project, now incorporating `analyze_results.py` with clear usage instructions, while ensuring the content is limited to replication and runtime guidance.

---

# Snake AI vs Human Performance Study

A comprehensive genetic algorithm-based AI system for playing Snake, with tools for comparing AI and human performance.

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

Run the full experiment workflow, including training, evaluation, and result analysis:

```bash
python experiment_manager.py
```

This script guides you through the entire process step-by-step.

### Method 2: Individual Components

#### Train AI Models
```bash
# Quick training (for testing)
python ga_train.py --quick-test

# Advanced training (longer runtime)
python ga_train.py
```

#### Evaluate Trained Models
```bash
python ga_snake.py --model experiment_data/best_ai/enhanced_model_final.pkl --display --games 10

```

#### Collect Human Performance Data including playing by adding users
```bash
python human_performance.py
```

#### Compare AI vs Human Performance
```bash
python compare_performance.py
```

#### Analyze Experiment Results
Use `analyze_results.py` to process experiment output files:

```bash
python analyze_results.py
```
Remember to change the file names in the script to match your output files.

**Note**: Requires libraries like `matplotlib` and `numpy` from `requirements.txt`.

### Method 3: Play as Human
```bash
python snake_game.py
```

Use arrow keys to control the snake.

## 📁 Project Structure

```
Snake_AI/
├── experiment_data/          # Experiment output directory
│   ├── ai_models/           # Trained AI models
│   ├── human_data/          # Human gameplay data
│   ├── results/             # Generated result files
│   └── best_ai/             # Best performing models
├── ga_models/               # Neural network implementations
├── experiment_manager.py    # Full experiment workflow
├── ga_train.py             # AI training script
├── ga_snake.py             # AI evaluation script
├── human_performance.py    # Human data collection
├── compare_performance.py  # Performance comparison
├── analyze_results.py      # Script to analyze experiment results
├── snake.py                # Core game logic
├── game_controller.py      # Human game interface
├── ga_controller.py        # AI game interface
├── continue_training.py    # Fine-tuning script
└── requirements.txt        # Python dependencies
```

## 🧪 Running Experiments

### Full Experiment
1. Run `experiment_manager.py` and follow the prompts:
   ```bash
   python experiment_manager.py
   ```
2. After generating results, analyze them with:
   ```bash
   python analyze_results.py --input <result-file> --output <report-file>
   ```

### Quick Test
```bash
# Train a quick model
python ga_train.py --quick-test

# Test it
python ga_snake.py --test --test-games 10
```