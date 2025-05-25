#!/usr/bin/env python
"""
Script for comparing AI and human performance in the Snake game.
Performs statistical analysis on performance data from both AI and human players.
"""
import json
import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent issues
import matplotlib.pyplot as plt
from scipy import stats
import pickle


class PerformanceAnalyzer:
    """Analyzes and compares AI vs human performance in the Snake game."""
    
    def __init__(self, ai_model_file='experiment_data/best_ai/enhanced_model_final.pkl', 
                 human_data_file='experiment_data/human_data/human_results.json',
                 ai_eval_games=None,
                 filter_scores=False):
        """
        Initialize the performance analyzer.
        
        Args:
            ai_model_file: Path to the AI model file to evaluate
            human_data_file: Path to the human results file
            ai_eval_games: Number of AI games to evaluate (overrides config if provided)
            filter_scores: Whether to filter out human games with very low scores that might be unfair
        """
        self.ai_model_file = ai_model_file
        self.human_data_file = human_data_file
        
        # Load experiment configuration to ensure consistent game count
        config_path = 'experiment_data/experiment_config.json'
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Use config value, but allow override through parameter
                self.ai_eval_games = ai_eval_games if ai_eval_games is not None else config.get('ai_games_per_evaluation', 34)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to parameter value or default if config file not found
            self.ai_eval_games = ai_eval_games if ai_eval_games is not None else 34
            print(f"Warning: Could not load experiment config. Using {self.ai_eval_games} AI evaluation games.")
        
        self.filter_scores = filter_scores
        
        # Load human data
        self.human_data = self._load_human_data()
    
    def _load_human_data(self):
        """Load human data from file, with better error handling."""
        try:
            print(f"Attempting to load human data from: {self.human_data_file}")
            with open(self.human_data_file, 'r') as f:
                human_data = json.load(f)
                # Basic validation to ensure it's in the expected format
                if 'sessions' not in human_data:
                    print(f"Error: The human data file doesn't contain expected 'sessions' key")
                    return None
                print(f"Successfully loaded human data with {len(human_data['sessions'])} sessions")
                return human_data
        except FileNotFoundError:
            print(f"Error: Human data file not found at {self.human_data_file}")
        except json.JSONDecodeError:
            print(f"Error: Human data file contains invalid JSON")
        except Exception as e:
            print(f"Error loading human data: {str(e)}")
        return None
    
    def run_ai_evaluation(self):
        """
        Run AI evaluation games.
        
        Returns:
            dict: AI performance results
        """
        from ga_snake import run_best_model
        
        print(f"Running {self.ai_eval_games} AI evaluation games...")
        avg_score, scores = run_best_model(
            model_file=self.ai_model_file,
            num_games=self.ai_eval_games,  # Use the value from config
            max_steps=2000,
            display=False,
            save_stats=True,
            output_dir="experiment_data/results"
        )
        
        # Load saved statistics for more detailed results
        output_dir = "experiment_data/results"
        model_name = os.path.basename(self.ai_model_file).split('.')[0]
        stats_file = f"{output_dir}/{model_name}_eval_{self.ai_eval_games}games.pkl"
        
        try:
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            # If stats file not found, use just the basic results
            stats = {
                'scores': scores,
                'avg_score': avg_score,
                'steps': [],  # We don't have steps data in this case
                'avg_steps': 0
            }
        
        return {
            'scores': stats['scores'],
            'steps': stats.get('steps', []),
            'avg_score': stats['avg_score'],
            'avg_steps': stats.get('avg_steps', 0),
            'max_score': max(stats['scores']) if stats['scores'] else 0
        }
    
    def get_human_performance(self):
        """
        Extract human performance data.
        
        Returns:
            dict: Human performance results
        """
        if not self.human_data:
            print("No human data available. Please check the path to the human data file.")
            return None
            
        all_scores = []
        all_steps = []
        
        for session in self.human_data.get('sessions', []):
            scores = session.get('scores', [])
            steps = session.get('steps', [])
            
            # Apply filtering if requested - exclude scores of 0 or 1 which might be from quick start
            if self.filter_scores:
                filtered_data = [(score, step) for score, step in zip(scores, steps) if score > 1]
                if filtered_data:
                    filtered_scores, filtered_steps = zip(*filtered_data)
                    all_scores.extend(filtered_scores)
                    all_steps.extend(filtered_steps)
            else:
                all_scores.extend(scores)
                all_steps.extend(steps)
        
        # If filtering, ensure we don't end up with too few games
        if self.filter_scores and len(all_scores) < 10:
            print("Warning: Less than 10 games after filtering. Using all human data instead.")
            # Try again without filtering
            self.filter_scores = False
            return self.get_human_performance()
        
        if not all_scores:
            print("Error: No valid human scores found in the data.")
            return None
            
        return {
            'scores': all_scores,
            'steps': all_steps,
            'avg_score': np.mean(all_scores) if all_scores else 0,
            'avg_steps': np.mean(all_steps) if all_steps else 0,
            'max_score': max(all_scores) if all_scores else 0
        }
    
    def generate_report(self):
        """
        Generate a comprehensive performance comparison report.
        
        Returns:
            dict: Comparison report
        """
        # Get AI performance
        ai_performance = self.run_ai_evaluation()
        
        # Get human performance
        human_performance = self.get_human_performance()
        
        if not human_performance:
            print("Error: No human data available. Please collect human data first.")
            return None
            
        # Balance the number of games if needed
        ai_scores = ai_performance['scores']
        human_scores = human_performance['scores']
        
        # Perform statistical comparison
        ai_avg = ai_performance['avg_score']
        human_avg = human_performance['avg_score']
        
        
        # Comparison result
        is_significant = p_value_t < 0.05
        better_performer = "AI" if ai_avg > human_avg else "Human"
        
        # Create the final report
        report = {
            'ai_performance': ai_performance,
            'human_performance': human_performance,
            'comparison': {
                'ai_avg_score': ai_avg,
                'human_avg_score': human_avg,
                'is_significant': bool(is_significant),
                'better_performer': better_performer
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print summary
        print("\n=== Performance Comparison ===")
        print(f"AI Performance: Avg Score = {ai_avg:.2f}, Games = {len(ai_scores)}")
        print(f"Human Performance: Avg Score = {human_avg:.2f}, Games = {len(human_scores)}")
        print(f"Difference: {ai_avg - human_avg:.2f} points")
        print(f"Statistical Significance (t-test): p = {p_value_t:.6f}")
        print(f"Statistical Significance (Mann-Whitney U): p = {p_value_u:.6f}")
        print(f"Result: {better_performer} performs {'significantly ' if is_significant else ''}better")
        
        if self.filter_scores:
            print(f"Note: {len(human_performance['scores'])} human games used after filtering out potentially unfair games")
            
        # Generate the detailed comparison visualization
        self.generate_multi_panel_visualization(ai_performance, human_performance)
        
        return report
        
    def generate_multi_panel_visualization(self, ai_performance, human_performance):
        """
        Generate a visualization comparing AI and human performance.
        
        Args:
            ai_performance: Dictionary with AI performance data
            human_performance: Dictionary with human performance data
        """
        # Create output directory if it doesn't exist
        vis_dir = 'experiment_data/results/visualizations'
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data
        ai_scores = ai_performance['scores']
        human_scores = human_performance['scores']
        ai_steps = ai_performance['steps']
        human_steps = human_performance['steps']
        
        # Create a single figure for the Steps vs Score scatter plot
        plt.figure(figsize=(10, 8))
        
        # Steps vs Score scatter plot
        plt.scatter(ai_steps, ai_scores, alpha=0.7, label='AI', color='skyblue')
        plt.scatter(human_steps, human_scores, alpha=0.7, label='Human', color='sandybrown')
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Steps vs Score Comparison', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save the figure
        output_file = os.path.join(vis_dir, 'steps_vs_score_comparison.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Steps vs Score comparison saved to: {output_file}")


def main():
    """Main function to run the performance comparison"""
    analyzer = PerformanceAnalyzer()
    
    while True:
        print("\nSnake Game Performance Analyzer")
        print("------------------------------")
        print("1. Run AI Evaluation")
        print("2. Extract Human Performance Data")
        print("3. Generate Report")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == '1':
            analyzer.run_ai_evaluation()
            
        elif choice == '2':
            human_data = analyzer.get_human_performance()
            if human_data:
                print("\nHuman Performance Summary:")
                print(f"Average Score: {human_data['avg_score']:.2f}")
                print(f"Average Steps: {human_data['avg_steps']:.2f}")
                print(f"Maximum Score: {human_data['max_score']}")
                
        elif choice == '3':
            report = analyzer.generate_report()
            if report:
                print("\nReport generated successfully.")
                
        elif choice == '4':
            print("Exiting Performance Analyzer.")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
