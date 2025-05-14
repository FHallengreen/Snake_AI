#!/usr/bin/env python
"""
Paper template generator for Snake AI vs. Human research project.
This script creates a LaTeX or Markdown template for a research paper
based on the experimental results.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

class PaperGenerator:
    """
    Generates a scientific paper template based on experimental results.
    """
    
    def __init__(self, results_dir='experiment_data/results', output_dir='paper'):
        """Initialize the paper generator."""
        self.results_dir = results_dir
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.report_data = None
        self.strategy_data = None
        
    def load_experimental_data(self):
        """Load experimental data from results files."""
        # Load comparison report
        report_path = os.path.join(self.results_dir, 'final_comparison_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                self.report_data = json.load(f)
            print(f"Loaded comparison report from {report_path}")
            
        # Load strategy analysis data
        strategy_path = os.path.join(self.results_dir, 'strategy_stats.json')
        if os.path.exists(strategy_path):
            with open(strategy_path, 'r') as f:
                self.strategy_data = json.load(f)
            print(f"Loaded strategy analysis from {strategy_path}")
            
        # Check if we have enough data
        if not self.report_data:
            print("Warning: No comparison report data found. Paper will have placeholders.")
            
        return True
        
    def generate_markdown_paper(self):
        """
        Generate a Markdown template for the research paper.
        """
        today = datetime.today().strftime('%Y-%m-%d')
        
        # Get result data if available
        if self.report_data:
            ai_score = self.report_data['ai_performance']['avg_score']
            human_score = self.report_data['human_performance']['avg_score']
            is_significant = self.report_data['comparison']['is_significant']
            better = self.report_data['comparison']['better_performer']
            t_pval = self.report_data['comparison']['t_test']['p_value']
            
            result_text = f"The results show that {better} performed better, with an average score of "
            if better == "AI":
                result_text += f"{ai_score:.2f} compared to {human_score:.2f} for human players. "
            else:
                result_text += f"{human_score:.2f} compared to {ai_score:.2f} for the AI. "
                
            result_text += f"This difference was {'statistically significant' if is_significant else 'not statistically significant'} (p={t_pval:.4f})."
        else:
            result_text = "[Results will be inserted here after experiments are completed]"
            
        # Strategy insights if available
        if self.strategy_data:
            ai_efficiency = self.strategy_data['AI']['food_efficiency']['mean']
            human_efficiency = self.strategy_data['Human']['food_efficiency']['mean']
            
            strategy_text = f"Analysis of gameplay strategies revealed that the AI had a food efficiency of {ai_efficiency:.4f} items per step, "
            strategy_text += f"compared to {human_efficiency:.4f} for human players."
        else:
            strategy_text = "[Strategy analysis will be inserted here after data collection]"
        
        # Create the paper template
        markdown = f"""# Comparing Genetic Algorithm-Trained Neural Networks with Human Performance in the Snake Game

**Date:** {today}

## Abstract

This study investigates the application of Genetic Algorithms (GA) to evolve neural networks capable of playing the classic Snake game, with performance compared to human players across various skill levels. The research explores whether evolutionary algorithms can produce gaming agents that outperform humans in decision-making speed, strategic planning, and overall score achievement. Using a controlled game environment with identical parameters for both AI and human players, this study quantifies performance differences through statistical analysis of game scores and survival duration. The findings contribute to our understanding of evolutionary algorithms' effectiveness in developing game-playing strategies that may exceed human capabilities in dynamic, constrained environments.

## 1. Introduction

Video games present well-defined environments for testing artificial intelligence approaches against human cognitive abilities. The Snake game, with its simple rules yet complex strategic requirements, provides an ideal testbed for comparing evolutionary algorithms with human decision-making processes. Genetic algorithms, inspired by natural selection, represent a promising approach to developing game-playing strategies without explicit programming.

This research implements a neural network whose weights are evolved through a genetic algorithm, allowing the AI to develop strategies for maximizing score while avoiding obstacles. The comparative performance between this evolved AI and human players yields insights into the strengths and limitations of evolutionary computation in game environments where decision speed, pattern recognition, and long-term planning are all critical factors.

### 1.1 Research Question

The primary research question addressed in this study is: Can a Genetic Algorithm train a neural network to play the Snake game more effectively than human players?

### 1.2 Hypothesis

- **Hypothesis (H1):** The GA-trained AI achieves a significantly higher average score in the Snake game than human players.
- **Null Hypothesis (H0):** There is no significant difference in the average scores between the GA-trained AI and human players.

## 2. Related Work

[This section would include a literature review of previous research in:
- Genetic algorithms for game AI
- Neural networks in game-playing agents
- Human vs. AI performance comparisons
- Snake game AI implementations]

## 3. Methods

### 3.1 Experimental Setup

This study employs a quantitative experimental research design comparing AI and human performance in identical game environments. The Snake game serves as the control environment, with performance measured through objective metrics including score (food items collected) and survival duration (steps taken before game termination).

### 3.2 Game Environment Implementation

The Snake game is implemented with the following controlled parameters:
- Grid size: 20×20 cells
- Snake movement: Up, Down, Left, Right (cardinal directions only)
- Game over conditions: Hitting the wall or the snake's own body
- Scoring: +1 for each food item collected

### 3.3 GA Model Development

The Genetic Algorithm training process includes:
- Neural network architecture: Input layer (15 neurons), Hidden layers (50, 50 neurons), Output layer (4 neurons)
- Input features: Distance to walls, food position relative to snake, danger detection in immediate vicinity, current direction
- Population size: Varied between 50 and 100 individuals
- Mutation rates: Varied between 0.01 and 0.1
- Selection method: Roulette wheel selection with elitism (top 10% preserved)
- Crossover: Weighted averaging of parent neural networks
- Fitness function: Based on score, survival steps, and food proximity
- Training duration: 100 generations with 3 evaluation games per individual

### 3.4 Human Performance Measurement Protocol

Human performance data was collected through:
- Participant recruitment with varying experience levels in gaming
- Collection of player demographics and self-reported experience level (1-10 scale)
- Controlled testing environment to minimize external variables
- Multiple play sessions per player (5 games minimum)
- Identical game parameters as used for AI training
- Recording of scores and game duration for each session
- Observation of gameplay strategies employed

### 3.5 Data Analysis Methodology

Statistical comparison between AI and human performance includes:
- Two-sample t-test for score comparison to determine statistical significance
- Mann-Whitney U test for non-parametric comparison to account for potential non-normal distributions
- Analysis of score and step distributions across participant experience levels
- Visualization of performance metrics using box plots and histograms
- Correlation analysis between experience level and performance difference with AI

## 4. Results

### 4.1 Overall Performance Comparison

{result_text}

### 4.2 Performance by Experience Level

[Analysis of how human player experience correlates with performance relative to AI]

### 4.3 Strategy Analysis

{strategy_text}

### 4.4 Statistical Tests

[Detailed results of statistical tests comparing AI and human performance]

## 5. Discussion

### 5.1 Interpretation of Results

[Interpretation of the performance comparison results, addressing the research question and hypothesis]

### 5.2 Strategy Differences

[Analysis of the different strategies employed by AI and human players]

### 5.3 Implications for AI and Game Design

[Discussion of implications for game AI development and design]

### 5.4 Limitations

[Limitations of the current study, including potential biases and constraints]

## 6. Conclusion

[Summary of findings, overall conclusions, and answer to the research question]

## 7. Future Work

[Potential directions for future research extending this work]

## References

[List of references and citations]

## Appendices

### Appendix A: Neural Network Architecture

[Technical details of the neural network architecture]

### Appendix B: Genetic Algorithm Parameters

[Complete list of GA parameters and settings]

### Appendix C: Human Player Demographics

[Summary of human player demographics and experience levels]
"""
        
        # Save the markdown file
        md_path = os.path.join(self.output_dir, 'paper.md')
        with open(md_path, 'w') as f:
            f.write(markdown)
            
        print(f"Markdown paper template generated: {md_path}")
        return md_path
    
    def generate_latex_paper(self):
        """
        Generate a LaTeX template for the research paper.
        """
        today = datetime.today().strftime('%Y-%m-%d')
        
        # Get result data if available
        if self.report_data:
            ai_score = self.report_data['ai_performance']['avg_score']
            human_score = self.report_data['human_performance']['avg_score']
            is_significant = self.report_data['comparison']['is_significant']
            better = self.report_data['comparison']['better_performer']
            t_pval = self.report_data['comparison']['t_test']['p_value']
            
            result_text = f"The results show that {better} performed better, with an average score of "
            if better == "AI":
                result_text += f"{ai_score:.2f} compared to {human_score:.2f} for human players. "
            else:
                result_text += f"{human_score:.2f} compared to {ai_score:.2f} for the AI. "
                
            result_text += f"This difference was {'statistically significant' if is_significant else 'not statistically significant'} (p={t_pval:.4f})."
        else:
            result_text = "[Results will be inserted here after experiments are completed]"
            
        # Create the LaTeX template
        latex = r"""\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{hyperref}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

\title{Comparing Genetic Algorithm-Trained Neural Networks with Human Performance in the Snake Game}

\author{\IEEEauthorblockN{Author Name}
\IEEEauthorblockA{Department of Computer Science\\
University Name\\
Email: author@example.com}}

\maketitle

\begin{abstract}
This study investigates the application of Genetic Algorithms (GA) to evolve neural networks capable of playing the classic Snake game, with performance compared to human players across various skill levels. The research explores whether evolutionary algorithms can produce gaming agents that outperform humans in decision-making speed, strategic planning, and overall score achievement. Using a controlled game environment with identical parameters for both AI and human players, this study quantifies performance differences through statistical analysis of game scores and survival duration. The findings contribute to our understanding of evolutionary algorithms' effectiveness in developing game-playing strategies that may exceed human capabilities in dynamic, constrained environments.
\end{abstract}

\begin{IEEEkeywords}
genetic algorithms, neural networks, game AI, human-AI comparison, evolutionary computation
\end{IEEEkeywords}

\section{Introduction}
Video games present well-defined environments for testing artificial intelligence approaches against human cognitive abilities. The Snake game, with its simple rules yet complex strategic requirements, provides an ideal testbed for comparing evolutionary algorithms with human decision-making processes. Genetic algorithms, inspired by natural selection, represent a promising approach to developing game-playing strategies without explicit programming.

This research implements a neural network whose weights are evolved through a genetic algorithm, allowing the AI to develop strategies for maximizing score while avoiding obstacles. The comparative performance between this evolved AI and human players yields insights into the strengths and limitations of evolutionary computation in game environments where decision speed, pattern recognition, and long-term planning are all critical factors.

\subsection{Research Question}
The primary research question addressed in this study is: Can a Genetic Algorithm train a neural network to play the Snake game more effectively than human players?

\subsection{Hypothesis}
\begin{itemize}
    \item \textbf{Hypothesis (H1):} The GA-trained AI achieves a significantly higher average score in the Snake game than human players.
    \item \textbf{Null Hypothesis (H0):} There is no significant difference in the average scores between the GA-trained AI and human players.
\end{itemize}

\section{Related Work}
[This section would include a literature review of previous research in:
\begin{itemize}
    \item Genetic algorithms for game AI
    \item Neural networks in game-playing agents
    \item Human vs. AI performance comparisons
    \item Snake game AI implementations
\end{itemize}]

\section{Methods}

\subsection{Experimental Setup}
This study employs a quantitative experimental research design comparing AI and human performance in identical game environments. The Snake game serves as the control environment, with performance measured through objective metrics including score (food items collected) and survival duration (steps taken before game termination).

\subsection{Game Environment Implementation}
The Snake game is implemented with the following controlled parameters:
\begin{itemize}
    \item Grid size: 20$\times$20 cells
    \item Snake movement: Up, Down, Left, Right (cardinal directions only)
    \item Game over conditions: Hitting the wall or the snake's own body
    \item Scoring: +1 for each food item collected
\end{itemize}

\subsection{GA Model Development}
The Genetic Algorithm training process includes:
\begin{itemize}
    \item Neural network architecture: Input layer (15 neurons), Hidden layers (50, 50 neurons), Output layer (4 neurons)
    \item Input features: Distance to walls, food position relative to snake, danger detection in immediate vicinity, current direction
    \item Population size: Varied between 50 and 100 individuals
    \item Mutation rates: Varied between 0.01 and 0.1
    \item Selection method: Roulette wheel selection with elitism (top 10\% preserved)
    \item Crossover: Weighted averaging of parent neural networks
    \item Fitness function: Based on score, survival steps, and food proximity
    \item Training duration: 100 generations with 3 evaluation games per individual
\end{itemize}

\subsection{Human Performance Measurement Protocol}
Human performance data was collected through:
\begin{itemize}
    \item Participant recruitment with varying experience levels in gaming
    \item Collection of player demographics and self-reported experience level (1-10 scale)
    \item Controlled testing environment to minimize external variables
    \item Multiple play sessions per player (5 games minimum)
    \item Identical game parameters as used for AI training
    \item Recording of scores and game duration for each session
    \item Observation of gameplay strategies employed
\end{itemize}

\subsection{Data Analysis Methodology}
Statistical comparison between AI and human performance includes:
\begin{itemize}
    \item Two-sample t-test for score comparison to determine statistical significance
    \item Mann-Whitney U test for non-parametric comparison to account for potential non-normal distributions
    \item Analysis of score and step distributions across participant experience levels
    \item Visualization of performance metrics using box plots and histograms
    \item Correlation analysis between experience level and performance difference with AI
\end{itemize}

\section{Results}

\subsection{Overall Performance Comparison}
""" + result_text + r"""

\subsection{Performance by Experience Level}
[Analysis of how human player experience correlates with performance relative to AI]

\subsection{Strategy Analysis}
[Analysis of different strategies employed by AI and human players]

\subsection{Statistical Tests}
[Detailed results of statistical tests comparing AI and human performance]

\section{Discussion}

\subsection{Interpretation of Results}
[Interpretation of the performance comparison results, addressing the research question and hypothesis]

\subsection{Strategy Differences}
[Analysis of the different strategies employed by AI and human players]

\subsection{Implications for AI and Game Design}
[Discussion of implications for game AI development and design]

\subsection{Limitations}
[Limitations of the current study, including potential biases and constraints]

\section{Conclusion}
[Summary of findings, overall conclusions, and answer to the research question]

\section{Future Work}
[Potential directions for future research extending this work]

\begin{thebibliography}{00}
\bibitem{ref1} Author, A., "Title of the Paper," Journal Name, vol. 1, no. 1, pp. 1-10, 2022.
\bibitem{ref2} Author, B., "Another Reference," Conference Name, pp. 100-110, 2021.
\end{thebibliography}

\end{document}
"""
        
        # Save the LaTeX file
        tex_path = os.path.join(self.output_dir, 'paper.tex')
        with open(tex_path, 'w') as f:
            f.write(latex)
            
        print(f"LaTeX paper template generated: {tex_path}")
        return tex_path

def main():
    """
    Main function to run the paper generator.
    """
    print("Snake AI vs. Human Paper Generator")
    print("--------------------------------")
    
    generator = PaperGenerator()
    
    while True:
        print("\nPaper Generator Options:")
        print("1. Load experimental data")
        print("2. Generate Markdown paper template")
        print("3. Generate LaTeX paper template")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == '1':
            generator.load_experimental_data()
            
        elif choice == '2':
            md_path = generator.generate_markdown_paper()
            print(f"Markdown paper template created at: {md_path}")
            
        elif choice == '3':
            tex_path = generator.generate_latex_paper()
            print(f"LaTeX paper template created at: {tex_path}")
            
        elif choice == '4':
            print("Exiting Paper Generator.")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
