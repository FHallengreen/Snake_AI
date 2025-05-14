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

## 2. AI Model Training

### 2.1. Train Multiple GA Models
Train multiple genetic algorithm models with different hyperparameters:

1. Run the experiment manager: `python experiment_manager.py`
2. Select option 2: "Prepare GA models"
3. When prompted, specify how many models to train (recommended: 3-5)
4. Wait for training to complete (may take 30+ minutes depending on your hardware)

This will create multiple models with different hyperparameter configurations.

### 2.2. Select Best AI Model
Evaluate and select the best performing model:

1. Run the experiment manager: `python experiment_manager.py`
2. Select option 3: "Select best model"
3. Wait for evaluation to complete

The system will automatically evaluate each model across multiple games and select the best one for the final experiment.

## 3. Human Data Collection

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

## 5. Generate Paper

### 5.1. Configure Paper Parameters

Edit the paper generator configuration if needed:

1. Open `paper_generator.py`
2. Review and adjust parameters like paper title, author names, and sections

### 5.2. Generate Research Paper

Run the paper generator to create a draft of your research paper:

```bash
python paper_generator.py
```

This will create a LaTeX document and corresponding PDF in the results directory.

### 5.3. Review and Finalize Paper

1. Review the generated paper
2. Add any additional insights or interpretations
3. Edit sections as needed
4. Regenerate the paper with your changes

## 6. Extended Analysis (Optional)

### 6.1. Strategy Analysis
Run the strategy analyzer to examine decision-making patterns:

```bash
python strategy_analyzer.py
```

### 6.2. Performance Plotting
Generate additional custom plots:

```bash
python plot_results.py
```

## 7. Project Completion

### 7.1. Final Check
Ensure all components of the project are complete:
- AI models trained and evaluated
- Human data collected (minimum 30 games)
- Comparison analysis completed
- Paper generated and reviewed

### 7.2. Backup Data
Create a backup of your experiment data:

```bash
cp -r experiment_data/ experiment_data_backup_$(date +%Y%m%d)/
```

### 7.3. Document Issues and Limitations
Add a section to your paper or create a separate document noting:
- Any limitations in your methodology
- Potential biases in data collection
- Ideas for future improvements

## Troubleshooting

### Common Issues

#### Model Training Errors
If you encounter errors during model training:
1. Check that all dependencies are installed
2. Verify that the GA parameters are within reasonable ranges
3. Try reducing the population size or number of generations

#### Human Data Collection Issues
If the human game interface is not working properly:
1. Verify pygame is installed correctly
2. Check that the screen resolution is compatible
3. Try running in a window instead of fullscreen

#### Analysis Errors
If you encounter errors during analysis:
1. Verify that sufficient data has been collected
2. Check that the best model file exists
3. Ensure all data files are in the expected format

For any persistent issues, check the project documentation or seek assistance.
