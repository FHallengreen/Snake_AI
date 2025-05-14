#!/usr/bin/env python
"""
Script for comparing AI and human performance in the Snake game.
Performs statistical analysis on performance data from both AI and human players.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
import pickle
import os
from snake import SnakeGame
from ga_models.ga_simple import SimpleModel


class PerformanceAnalyzer:
    def __init__(self, ai_model_file='best_model.pkl', human_data_file='human_results.json'):
        self.ai_model_file = ai_model_file
        self.human_data_file = human_data_file
        self.ai_data = None
        self.human_data = None
        
    def load_data(self):
        """Load AI and human performance data"""
        # Load AI model and data
        if os.path.exists(self.ai_model_file):
            with open(self.ai_model_file, 'rb') as f:
                self.ai_data = pickle.load(f)
            print("AI data loaded successfully")
        else:
            print(f"AI model file {self.ai_model_file} not found.")
            return False
            
        # Load human performance data
        if os.path.exists(self.human_data_file):
            try:
                with open(self.human_data_file, 'r') as f:
                    self.human_data = json.load(f)
                print("Human data loaded successfully")
            except json.JSONDecodeError:
                print(f"Error reading human data file {self.human_data_file}.")
                return False
        else:
            print(f"Human data file {self.human_data_file} not found.")
            return False
            
        return True
        
    def evaluate_ai_performance(self, num_games=30):
        """Run the AI model to generate fresh performance data"""
        if not os.path.exists(self.ai_model_file):
            print(f"AI model file {self.ai_model_file} not found.")
            return None
            
        # Load the AI model
        with open(self.ai_model_file, 'rb') as f:
            model_data = pickle.load(f)
            
        # Create the model and set its DNA
        model = SimpleModel(dims=(15, 50, 50, 4))
        model.DNA = model_data['DNA']
        
        # Run games and collect scores
        game = SnakeGame(xsize=20, ysize=20)
        scores = []
        steps_list = []
        
        print(f"Evaluating AI model over {num_games} games...")
        
        for i in range(num_games):
            score, steps = game.run(model=model, max_steps=2000, display=False)
            scores.append(score)
            steps_list.append(steps)
            print(f"Game {i+1}/{num_games}: Score = {score}, Steps = {steps}")
            
        # Calculate statistics
        avg_score = np.mean(scores)
        avg_steps = np.mean(steps)
        max_score = np.max(scores)
        
        print("\nAI Performance Summary:")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Maximum Score: {max_score}")
        
        return {
            'scores': scores,
            'steps': steps_list,
            'avg_score': float(avg_score),
            'avg_steps': float(avg_steps),
            'max_score': int(max_score)
        }
        
    def extract_human_performance_data(self):
        """Extract performance data from all human players"""
        if not self.human_data:
            print("Human data not loaded.")
            return None
            
        all_scores = []
        all_steps = []
        
        for session in self.human_data['sessions']:
            all_scores.extend(session['scores'])
            all_steps.extend(session['steps'])
            
        if not all_scores:
            print("No human performance data found.")
            return None
            
        return {
            'scores': all_scores,
            'steps': all_steps,
            'avg_score': float(np.mean(all_scores)),
            'avg_steps': float(np.mean(all_steps)),
            'max_score': int(np.max(all_scores))
        }
        
    def compare_performance(self, ai_data=None, human_data=None):
        """
        Compare AI performance against human performance
        
        Args:
            ai_data: AI performance data (if None, will use existing data)
            human_data: Human performance data (if None, will extract from loaded data)
            
        Returns:
            dict: Comparison results
        """
        if ai_data is None:
            # Use existing AI data or generate new data
            if self.ai_data and 'scores' in self.ai_data:
                ai_data = self.ai_data
            else:
                ai_data = self.evaluate_ai_performance()
                
        if human_data is None:
            # Extract human performance data
            human_data = self.extract_human_performance_data()
            
        if not ai_data or not human_data:
            print("Missing performance data for comparison.")
            return None
            
        # Get the scores for comparison
        ai_scores = ai_data['scores'] if isinstance(ai_data['scores'], list) else [ai_data['scores']]
        human_scores = human_data['scores']
        
        # Compute statistical tests
        try:
            t_stat, t_pval = ttest_ind(ai_scores, human_scores)
            u_stat, u_pval = mannwhitneyu(ai_scores, human_scores)
            
            # Determine statistical significance (alpha = 0.05)
            is_significant = t_pval < 0.05 or u_pval < 0.05
            better_performer = "AI" if np.mean(ai_scores) > np.mean(human_scores) else "Human"
            
            comparison = {
                'ai_avg_score': float(np.mean(ai_scores)),
                'human_avg_score': float(np.mean(human_scores)),
                't_test': {'statistic': float(t_stat), 'p_value': float(t_pval)},
                'u_test': {'statistic': float(u_stat), 'p_value': float(u_pval)},
                'is_significant': bool(is_significant),
                'better_performer': better_performer
            }
            
            return comparison
            
        except Exception as e:
            print(f"Error during statistical comparison: {e}")
            return None
            
    def plot_comparison(self, ai_data=None, human_data=None):
        """
        Plot comparison between AI and human performance
        
        Args:
            ai_data: AI performance data (if None, will use existing data)
            human_data: Human performance data (if None, will extract from loaded data)
        """
        if ai_data is None:
            if self.ai_data and 'scores' in self.ai_data:
                ai_data = self.ai_data
            else:
                ai_data = self.evaluate_ai_performance()
                
        if human_data is None:
            human_data = self.extract_human_performance_data()
            
        if not ai_data or not human_data:
            print("Missing performance data for plotting.")
            return
            
        ai_scores = ai_data['scores'] if isinstance(ai_data['scores'], list) else [ai_data['scores']]
        human_scores = human_data['scores']
        
        plt.figure(figsize=(12, 8))
        
        # Score distribution
        plt.subplot(2, 2, 1)
        plt.hist(ai_scores, alpha=0.5, label="AI", bins=10)
        plt.hist(human_scores, alpha=0.5, label="Human", bins=10)
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.title("Score Distribution")
        plt.legend()
        
        # Average score comparison
        plt.subplot(2, 2, 2)
        avg_ai = np.mean(ai_scores)
        avg_human = np.mean(human_scores)
        std_ai = np.std(ai_scores)
        std_human = np.std(human_scores)
        
        plt.bar(["AI", "Human"], [avg_ai, avg_human], yerr=[std_ai, std_human], capsize=10)
        plt.ylabel("Average Score")
        plt.title("Average Score Comparison")
        
        # Steps comparison
        plt.subplot(2, 2, 3)
        ai_steps = ai_data['steps'] if isinstance(ai_data['steps'], list) else [ai_data['steps']]
        human_steps = human_data['steps']
        
        plt.hist(ai_steps, alpha=0.5, label="AI", bins=10)
        plt.hist(human_steps, alpha=0.5, label="Human", bins=10)
        plt.xlabel("Steps")
        plt.ylabel("Frequency")
        plt.title("Steps Distribution")
        plt.legend()
        
        # Score vs. Steps
        plt.subplot(2, 2, 4)
        plt.scatter(ai_steps, ai_scores, alpha=0.6, label="AI")
        plt.scatter(human_steps, human_scores, alpha=0.6, label="Human")
        plt.xlabel("Steps")
        plt.ylabel("Score")
        plt.title("Score vs. Steps")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('ai_vs_human_comparison.png')
        plt.show()
        
    def generate_report(self):
        """Generate a comprehensive comparison report"""
        if not self.load_data():
            return
            
        # Evaluate AI with fresh runs
        ai_perf = self.evaluate_ai_performance(num_games=30)
        human_perf = self.extract_human_performance_data()
        
        if not ai_perf or not human_perf:
            print("Cannot generate report: missing performance data")
            return
            
        # Compare performance
        comparison = self.compare_performance(ai_perf, human_perf)
        
        if not comparison:
            print("Error during comparison")
            return
            
        # Create report
        report = {
            'ai_performance': ai_perf,
            'human_performance': human_perf,
            'comparison': comparison,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save report
        with open('performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print("\n===== Performance Report =====")
        print(f"AI Average Score: {ai_perf['avg_score']:.2f}")
        print(f"Human Average Score: {human_perf['avg_score']:.2f}")
        print(f"Difference: {ai_perf['avg_score'] - human_perf['avg_score']:.2f}")
        print(f"Statistical Significance: {'Yes' if comparison['is_significant'] else 'No'}")
        print(f"Better Performer: {comparison['better_performer']}")
        print(f"T-test p-value: {comparison['t_test']['p_value']:.4f}")
        print(f"U-test p-value: {comparison['u_test']['p_value']:.4f}")
        print("===============================")
        
        # Generate visual report
        self.plot_comparison(ai_perf, human_perf)
        
        return report


def main():
    """Main function to run the performance comparison"""
    analyzer = PerformanceAnalyzer()
    
    while True:
        print("\nSnake Game Performance Analyzer")
        print("------------------------------")
        print("1. Load Data")
        print("2. Evaluate AI Performance")
        print("3. Extract Human Performance Data")
        print("4. Compare Performance")
        print("5. Generate Report with Visualization")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ")
        
        if choice == '1':
            analyzer.load_data()
            
        elif choice == '2':
            analyzer.evaluate_ai_performance()
            
        elif choice == '3':
            human_data = analyzer.extract_human_performance_data()
            if human_data:
                print("\nHuman Performance Summary:")
                print(f"Average Score: {human_data['avg_score']:.2f}")
                print(f"Average Steps: {human_data['avg_steps']:.2f}")
                print(f"Maximum Score: {human_data['max_score']}")
                
        elif choice == '4':
            result = analyzer.compare_performance()
            if result:
                print("\nComparison Results:")
                print(f"AI Average Score: {result['ai_avg_score']:.2f}")
                print(f"Human Average Score: {result['human_avg_score']:.2f}")
                print(f"T-test p-value: {result['t_test']['p_value']:.4f}")
                print(f"U-test p-value: {result['u_test']['p_value']:.4f}")
                print(f"Statistically Significant: {'Yes' if result['is_significant'] else 'No'}")
                print(f"Better Performer: {result['better_performer']}")
                
        elif choice == '5':
            analyzer.generate_report()
            
        elif choice == '6':
            print("Exiting Performance Analyzer.")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
