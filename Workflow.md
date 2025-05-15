# Snake AI vs. Human Comparison Workflow

This document provides detailed step-by-step instructions for completing the Snake AI vs. Human comparison study from start to finish. Follow these steps in sequence to ensure proper data collection, model training, analysis, and paper generation.

## 1. Environment Setup

### 1.1. Verify Installation
First, ensure that all required packages are installed:

```bash
pip install -r requirements.txt
```

### 1.2. Initialize Experiment Structure
Run the experiment manager to create the necessary directory structure:

```bash
python experiment_manager.py
```

Select option 1 from the menu to "Setup experiment (create directory structure)".

## 2. AI Model Training (COMPLETED)

### 2.1. Train Multiple GA Models ✓
We've trained multiple genetic algorithm models with different hyperparameters:

```bash
python experiment_manager.py
```

Selected option 2 and chosen "y" for advanced training with these parameters:
- Population sizes: 150
- Mutation rates: 0.032-0.035
- Games per evaluation: 5
- Neural network: (21, 128, 64, 32, 4)

### 2.2. Select Best AI Model ✓
We've evaluated and selected the best model:

```bash
python experiment_manager.py
```

Selected option 3, which:
- Ran 10 evaluation games for each model
- Selected the model with highest average score (46.80)
- Copied the best model to experiment_data/best_model.pkl

### 2.3. Continue Training from Best Model ✓
We've performed continued training:

```bash
python continue_training.py --pop-size 150 --mut-rate 0.032 --games 5 --generations 150
```

This yielded:
- Improved training score: 60.14 (from generation 88)
- Better maximum game performance
- Multiple enhanced models that were evaluated

### 2.4. Final Model Selection ✓
We've conducted final validation:
- Evaluated all enhanced models with 10 games each
- Selected enhanced_model_3.pkl as the best (score 42.90)
- Concluded that performance has effectively plateaued

## 3. Human Data Collection (NEXT STEP)

### 3.1. Recruit Participants
Recruit 8-12 human participants with varying levels of gaming experience:
- Beginners (no/little Snake game experience)
- Intermediate (some experience with Snake games)
- Advanced (experienced Snake game players)

Document participant information including:
- Age
- Gaming experience level
- Previous Snake game experience

### 3.2. Collect Human Performance Data
For each participant:

1. Run the human performance tracker:
   ```bash
   python human_performance.py
   ```

2. For each participant session:
   - Enter participant ID (create a new ID for first-time participants)
   - Enter participant information when prompted
   - Have participant play the required number of games (typically 3-5 games per session)
   - Save results when finished

3. Continue until you have collected data from all participants
   - Aim for at least 30 total games across all participants (as configured in the experiment)

### 3.3. Monitor Data Collection Progress
Check the progress of human data collection:

1. Run the experiment manager: `python experiment_manager.py`
2. Select option 4: "Collect human data"

This will show you the current progress and how many more games are needed.

## 4. Analysis and Comparison

### 4.1. Run Final Experiment
Execute the comparison between AI and human performance:

1. Run the experiment manager: `python experiment_manager.py`
2. Select option 5: "Run final experiment"

This will:
- Run the best AI model through multiple games
- Compare AI performance with human performance
- Generate statistical analysis
- Create visualizations

### 4.2. Analyze Results
Review the generated analysis in the `experiment_data/results/` directory:

1. Examine the final comparison report (`final_comparison_report.json`)
2. Review visualizations in the `visualizations/` subfolder:
   - Score distributions
   - Experience level vs. performance
   - Learning curves
   - Decision patterns

## 5. Paper Generation

### 5.1. Document Statistical Findings
Based on the comparative analysis, document:
- Whether AI outperforms humans (with statistical significance)
- How AI performance relates to human experience levels
- Differences in gameplay strategies and patterns
- Any identified limitations in the AI approach

### 5.2. Generate Research Paper
Create a comprehensive research paper with:
- Abstract summarizing the findings
- Introduction establishing the research context
- Methods section detailing both AI and human data collection
- Results presenting quantitative comparisons
- Discussion interpreting the findings
- Conclusion summarizing the research contribution

### 5.3. Visual Documentation
Include visualizations demonstrating:
- Performance distributions for AI vs humans
- Learning curves for both AI training and humans across multiple sessions
- Strategy comparisons (heatmaps, path efficiency, time-to-decision)

## 6. Extended Analysis (Optional)

### 6.1. Strategy Analysis
For deeper insights, analyze specific gameplay differences:
- How AI and humans handle "trapped" scenarios
- Food collection efficiency (steps per food item)
- Long-term planning vs. reactive decision making
- Different failure modes between AI and humans

### 6.2. Performance Predictions
If data permits, build predictive models of:
- Human performance based on experience level
- When and why humans outperform the AI (or vice versa)
- How much training human players would need to match the AI

## 7. Project Completion

### 7.1. Final Documentation
Ensure all aspects are documented:
- Training methodology and parameters
- Neural network architecture details
- Human data collection methodology
- Statistical analysis procedures
- Conclusions and implications

### 7.2. Data Preservation
Archive all project data:
- Trained models (best and enhanced versions)
- Raw human performance data
- Analysis scripts and results
- Generated visualizations

### 7.3. Future Directions
Identify promising avenues for future research:
- Alternative neural network architectures
- Different training approaches
- More complex game environments
- Transfer learning to similar games
