#!/usr/bin/env python
"""
Enhanced analysis module for Snake AI vs. Human comparison.
Provides deeper insights into gameplay strategies and patterns.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr
import json
import os
from collections import defaultdict
import pickle

class StrategyAnalyzer:
    """
    Analyzes gameplay strategies and patterns from AI and human players.
    """
    
    def __init__(self, results_dir='experiment_data/results'):
        """Initialize the strategy analyzer."""
        self.results_dir = results_dir
        self.viz_dir = os.path.join(results_dir, 'visualizations')
        
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
            
        self.report_data = None
        self.ai_gameplay_data = []
        self.human_gameplay_data = []
        
    def load_report_data(self, report_file='final_comparison_report.json'):
        """Load the comparison report data."""
        report_path = os.path.join(self.results_dir, report_file)
        
        if not os.path.exists(report_path):
            print(f"Report file {report_path} not found.")
            return False
            
        with open(report_path, 'r') as f:
            self.report_data = json.load(f)
            
        print(f"Report data loaded from {report_path}")
        return True
        
    def collect_gameplay_data(self, num_ai_games=10, num_human_games=None):
        """
        Collect detailed gameplay data for strategy analysis.
        
        Args:
            num_ai_games: Number of AI games to record
            num_human_games: Number of human games to analyze (if None, use all available)
        """
        from snake import SnakeGame
        from ga_models.ga_simple import SimpleModel
        
        # 1. Collect AI gameplay data
        print("Collecting AI gameplay data...")
        
        # Load the best AI model
        best_model_path = "experiment_data/best_model.pkl"
        if not os.path.exists(best_model_path):
            best_model_path = "best_model.pkl"
            
        if not os.path.exists(best_model_path):
            print("Best model file not found. Cannot collect AI gameplay data.")
            return
            
        with open(best_model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        model = SimpleModel(dims=(15, 50, 50, 4))
        model.DNA = model_data['DNA']
        
        # Run games with detailed data recording
        for game_num in range(num_ai_games):
            print(f"Recording AI game {game_num+1}/{num_ai_games}")
            game_data = self._record_game_data(model)
            if game_data:
                game_data['player_type'] = 'AI'
                self.ai_gameplay_data.append(game_data)
                
        # 2. Load human gameplay data
        print("Loading human gameplay data...")
        human_data_file = "human_results.json"
        
        if not os.path.exists(human_data_file):
            print(f"Human data file {human_data_file} not found.")
            return
            
        with open(human_data_file, 'r') as f:
            human_data = json.load(f)
            
        # Find session data with the most detailed information
        detailed_sessions = [s for s in human_data['sessions'] if 'detailed_data' in s]
        
        if not detailed_sessions:
            print("No detailed human gameplay data found.")
            print("Please run human games with data recording enabled.")
            return
            
        # Limit to requested number if specified
        if num_human_games is not None:
            detailed_sessions = detailed_sessions[:num_human_games]
            
        for session in detailed_sessions:
            for game_data in session['detailed_data']:
                game_data['player_type'] = 'Human'
                game_data['player_id'] = session['player_id']
                game_data['experience'] = human_data['players'][session['player_id']]['experience_level']
                self.human_gameplay_data.append(game_data)
                
        print(f"Collected {len(self.ai_gameplay_data)} AI games and {len(self.human_gameplay_data)} human games")
        
    def _record_game_data(self, model):
        """
        Record detailed data from a single game.
        
        Args:
            model: The AI model to use
        
        Returns:
            dict: Detailed game data
        """
        from snake import SnakeGame
        
        # Create a game with data recording
        game = SnakeGame(xsize=20, ysize=20)
        
        # Initialize data structures
        moves = []
        food_positions = []
        snake_positions = []
        danger_states = []
        
        # Start with recording the initial food position
        food_positions.append((game.food.p.x, game.food.p.y))
        
        # Run the game with data recording
        score, steps = game.run(
            model=model,
            max_steps=2000,
            display=False,
            record_data=True,
            moves=moves,
            food_positions=food_positions,
            snake_positions=snake_positions,
            danger_states=danger_states
        )
        
        # Check if we have meaningful data
        if score == 0 and steps < 100:
            # Game ended too quickly, not useful for analysis
            return None
            
        # Calculate metrics
        food_efficiency = score / steps if steps > 0 else 0
        
        # Calculate average distance to food over time
        distances = []
        for i in range(min(len(snake_positions), len(food_positions))):
            snake_head = snake_positions[i][0]  # First position is the head
            food_pos = food_positions[i]
            manhattan_dist = abs(snake_head[0] - food_pos[0]) + abs(snake_head[1] - food_pos[1])
            distances.append(manhattan_dist)
            
        avg_distance = np.mean(distances) if distances else 0
        
        # Create the game data dictionary
        game_data = {
            'score': score,
            'steps': steps,
            'food_efficiency': food_efficiency,
            'avg_distance_to_food': avg_distance,
            'moves': moves,
            'food_positions': food_positions,
            'snake_positions': snake_positions,
            'danger_states': danger_states
        }
        
        return game_data
        
    def analyze_strategies(self):
        """
        Perform strategy analysis on the collected gameplay data.
        """
        if not self.ai_gameplay_data or not self.human_gameplay_data:
            print("No gameplay data available. Run collect_gameplay_data first.")
            return
            
        # Create summary dataframes
        ai_df = pd.DataFrame([{
            'player_type': 'AI',
            'game_id': i,
            'score': game['score'],
            'steps': game['steps'],
            'food_efficiency': game['food_efficiency'],
            'avg_distance_to_food': game['avg_distance_to_food']
        } for i, game in enumerate(self.ai_gameplay_data)])
        
        human_df = pd.DataFrame([{
            'player_type': 'Human',
            'player_id': game['player_id'],
            'experience': game['experience'],
            'game_id': i,
            'score': game['score'],
            'steps': game['steps'],
            'food_efficiency': game['food_efficiency'],
            'avg_distance_to_food': game['avg_distance_to_food']
        } for i, game in enumerate(self.human_gameplay_data)])
        
        # Combined dataframe
        combined_df = pd.concat([ai_df, human_df], ignore_index=True)
        
        # Save the analysis data
        combined_df.to_csv(os.path.join(self.results_dir, 'strategy_analysis.csv'), index=False)
        
        # 1. Food efficiency comparison (score per step)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='player_type', y='food_efficiency', data=combined_df)
        plt.title('Food Collection Efficiency Comparison')
        plt.ylabel('Food Items per Step')
        plt.savefig(os.path.join(self.viz_dir, 'food_efficiency_comparison.png'))
        
        # 2. Distance to food comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='player_type', y='avg_distance_to_food', data=combined_df)
        plt.title('Average Distance to Food Comparison')
        plt.ylabel('Average Manhattan Distance')
        plt.savefig(os.path.join(self.viz_dir, 'distance_to_food_comparison.png'))
        
        # 3. Human experience vs. metrics
        if 'experience' in human_df.columns:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            sns.scatterplot(x='experience', y='score', data=human_df)
            plt.title('Experience vs. Score')
            
            plt.subplot(2, 2, 2)
            sns.scatterplot(x='experience', y='food_efficiency', data=human_df)
            plt.title('Experience vs. Food Efficiency')
            
            plt.subplot(2, 2, 3)
            sns.scatterplot(x='experience', y='steps', data=human_df)
            plt.title('Experience vs. Game Duration')
            
            plt.subplot(2, 2, 4)
            sns.scatterplot(x='experience', y='avg_distance_to_food', data=human_df)
            plt.title('Experience vs. Distance to Food')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'experience_metrics_correlation.png'))
            
        # 4. Movement pattern analysis
        ai_moves = []
        for game in self.ai_gameplay_data:
            ai_moves.extend(game['moves'])
            
        human_moves = []
        for game in self.human_gameplay_data:
            human_moves.extend(game['moves'])
            
        # Count movement directions
        ai_move_counts = defaultdict(int)
        human_move_counts = defaultdict(int)
        
        for move in ai_moves:
            ai_move_counts[move] += 1
            
        for move in human_moves:
            human_move_counts[move] += 1
            
        # Normalize counts
        total_ai_moves = sum(ai_move_counts.values())
        total_human_moves = sum(human_move_counts.values())
        
        directions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
        
        ai_move_pct = [ai_move_counts[d]/total_ai_moves*100 if total_ai_moves > 0 else 0 for d in directions]
        human_move_pct = [human_move_counts[d]/total_human_moves*100 if total_human_moves > 0 else 0 for d in directions]
        
        # Plot movement distribution
        plt.figure(figsize=(10, 6))
        x = np.arange(len(directions))
        width = 0.35
        
        plt.bar(x - width/2, ai_move_pct, width, label='AI')
        plt.bar(x + width/2, human_move_pct, width, label='Human')
        
        plt.xlabel('Movement Direction')
        plt.ylabel('Percentage of Moves')
        plt.title('Movement Direction Distribution')
        plt.xticks(x, directions)
        plt.legend()
        
        plt.savefig(os.path.join(self.viz_dir, 'movement_distribution.png'))
        
        # 5. Statistical analysis
        print("\n=== Strategy Statistical Analysis ===")
        
        # Food efficiency
        ai_efficiency = ai_df['food_efficiency'].values
        human_efficiency = human_df['food_efficiency'].values
        t_stat, p_val = ttest_ind(ai_efficiency, human_efficiency)
        
        print(f"Food Efficiency T-test: t={t_stat:.2f}, p={p_val:.4f}")
        print(f"AI Mean: {np.mean(ai_efficiency):.4f}, Human Mean: {np.mean(human_efficiency):.4f}")
        
        # Distance to food
        ai_distance = ai_df['avg_distance_to_food'].values
        human_distance = human_df['avg_distance_to_food'].values
        t_stat, p_val = ttest_ind(ai_distance, human_distance)
        
        print(f"Distance to Food T-test: t={t_stat:.2f}, p={p_val:.4f}")
        print(f"AI Mean: {np.mean(ai_distance):.4f}, Human Mean: {np.mean(human_distance):.4f}")
        
        # Save summary statistics
        summary_stats = {
            'AI': {
                'score': {
                    'mean': float(ai_df['score'].mean()),
                    'std': float(ai_df['score'].std()),
                    'min': int(ai_df['score'].min()),
                    'max': int(ai_df['score'].max())
                },
                'steps': {
                    'mean': float(ai_df['steps'].mean()),
                    'std': float(ai_df['steps'].std())
                },
                'food_efficiency': {
                    'mean': float(ai_df['food_efficiency'].mean()),
                    'std': float(ai_df['food_efficiency'].std())
                },
                'avg_distance_to_food': {
                    'mean': float(ai_df['avg_distance_to_food'].mean()),
                    'std': float(ai_df['avg_distance_to_food'].std())
                }
            },
            'Human': {
                'score': {
                    'mean': float(human_df['score'].mean()),
                    'std': float(human_df['score'].std()),
                    'min': int(human_df['score'].min()),
                    'max': int(human_df['score'].max())
                },
                'steps': {
                    'mean': float(human_df['steps'].mean()),
                    'std': float(human_df['steps'].std())
                },
                'food_efficiency': {
                    'mean': float(human_df['food_efficiency'].mean()),
                    'std': float(human_df['food_efficiency'].std())
                },
                'avg_distance_to_food': {
                    'mean': float(human_df['avg_distance_to_food'].mean()),
                    'std': float(human_df['avg_distance_to_food'].std())
                }
            }
        }
        
        with open(os.path.join(self.results_dir, 'strategy_stats.json'), 'w') as f:
            json.dump(summary_stats, f, indent=2)
            
        print(f"Analysis results saved to {self.results_dir}")
        
    def generate_heatmaps(self):
        """
        Generate positional heatmaps to visualize movement patterns.
        """
        if not self.ai_gameplay_data or not self.human_gameplay_data:
            print("No gameplay data available. Run collect_gameplay_data first.")
            return
            
        grid_size = 20
        
        # Initialize heatmaps
        ai_heatmap = np.zeros((grid_size, grid_size))
        human_heatmap = np.zeros((grid_size, grid_size))
        
        # Populate AI heatmap
        for game in self.ai_gameplay_data:
            for positions in game['snake_positions']:
                head_x, head_y = positions[0]  # First position is the head
                if 0 <= head_x < grid_size and 0 <= head_y < grid_size:
                    ai_heatmap[head_y, head_x] += 1  # Note: y is first index for 2D array
        
        # Populate human heatmap
        for game in self.human_gameplay_data:
            for positions in game['snake_positions']:
                head_x, head_y = positions[0]  # First position is the head
                if 0 <= head_x < grid_size and 0 <= head_y < grid_size:
                    human_heatmap[head_y, head_x] += 1
                    
        # Normalize heatmaps
        ai_heatmap = ai_heatmap / ai_heatmap.sum() if ai_heatmap.sum() > 0 else ai_heatmap
        human_heatmap = human_heatmap / human_heatmap.sum() if human_heatmap.sum() > 0 else human_heatmap
        
        # Plot heatmaps
        plt.figure(figsize=(16, 7))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(ai_heatmap, cmap='viridis')
        plt.title('AI Movement Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(human_heatmap, cmap='viridis')
        plt.title('Human Movement Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'movement_heatmaps.png'))
        
        # Calculate difference heatmap
        diff_heatmap = ai_heatmap - human_heatmap
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff_heatmap, cmap='coolwarm', center=0)
        plt.title('Movement Pattern Difference (AI - Human)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        plt.savefig(os.path.join(self.viz_dir, 'movement_difference_heatmap.png'))
        

def main():
    """
    Main function to run the strategy analysis.
    """
    print("Snake Game Strategy Analysis")
    print("--------------------------")
    
    analyzer = StrategyAnalyzer()
    
    while True:
        print("\nAnalysis Options:")
        print("1. Load report data")
        print("2. Collect gameplay data")
        print("3. Analyze strategies")
        print("4. Generate heatmaps")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == '1':
            analyzer.load_report_data()
            
        elif choice == '2':
            try:
                ai_games = int(input("Number of AI games to record (default: 10): ") or "10")
                human_games = input("Number of human games to analyze (default: all): ")
                human_games = int(human_games) if human_games else None
                
                analyzer.collect_gameplay_data(num_ai_games=ai_games, num_human_games=human_games)
            except ValueError:
                print("Invalid input. Using default values.")
                analyzer.collect_gameplay_data()
                
        elif choice == '3':
            analyzer.analyze_strategies()
            
        elif choice == '4':
            analyzer.generate_heatmaps()
            
        elif choice == '5':
            print("Exiting Strategy Analyzer.")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
