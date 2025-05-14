#!/usr/bin/env python
"""
Experimental design and data management for Snake AI vs. Human comparison study.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

class ExperimentManager:
    """
    Manages experimental design, data collection, and organization for the
    Snake AI vs Human comparison study.
    """
    
    def __init__(self, experiment_dir='experiment_data'):
        """
        Initialize the experiment manager.
        
        Args:
            experiment_dir: Directory to store experiment data
        """
        self.experiment_dir = experiment_dir
        self.ai_models_dir = os.path.join(experiment_dir, 'ai_models')
        self.human_data_dir = os.path.join(experiment_dir, 'human_data')
        self.results_dir = os.path.join(experiment_dir, 'results')
        
        # Create directories if they don't exist
        for directory in [self.experiment_dir, self.ai_models_dir, 
                         self.human_data_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        # Experiment configuration
        self.config = {
            'ai_games_per_evaluation': 30,
            'human_games_required': 30,
            'game_parameters': {
                'grid_size': 20,
                'max_steps': 2000
            }
        }
        
    def setup_experiment(self):
        """
        Initialize experiment structure and save configuration.
        """
        # Save configuration
        config_path = os.path.join(self.experiment_dir, 'experiment_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Experiment setup created at {self.experiment_dir}")
        print(f"Configuration saved to {config_path}")
        
    def prepare_ga_models(self, models_to_evaluate=3, advanced_training=False):
        """
        Prepare multiple GA models with different hyperparameters for evaluation.
        
        Args:
            models_to_evaluate: Number of different models to prepare
            advanced_training: Whether to use advanced training (more generations, better params)
        """
        from ga_train import run_ga_experiment
        
        # Check if user wants advanced training
        if advanced_training:
            print("\n=== Using Advanced Training Parameters ===")
            print("This will take longer but produce better models")
            
            # Reduced generations from 350 to 200, as data shows major improvements happen by gen 200
            configs = [
                {'pop_size': 150, 'mut_rate': 0.03, 'generations': 200, 'games': 5},  # Best-performing config
                {'pop_size': 160, 'mut_rate': 0.035, 'generations': 180, 'games': 5}, # Slightly higher mutation
                {'pop_size': 170, 'mut_rate': 0.025, 'generations': 180, 'games': 5}  # Slightly lower mutation
            ][:models_to_evaluate]
        else:
            print("\n=== Using Standard Training Parameters ===")
            print("For better models, consider using advanced training")
            
            # Reduced generations for faster results
            configs = [
                {'pop_size': 100, 'mut_rate': 0.03, 'generations': 80, 'games': 3},
                {'pop_size': 120, 'mut_rate': 0.035, 'generations': 80, 'games': 3},
                {'pop_size': 130, 'mut_rate': 0.025, 'generations': 80, 'games': 3}
            ][:models_to_evaluate]
        
        for i, config in enumerate(configs):
            print(f"\nTraining model {i+1}/{len(configs)}")
            print(f"Parameters: {config}")
            
            # Train model with all config parameters
            stats, best_model, best_fitness, best_score, best_steps, best_gen = run_ga_experiment(
                pop_size=config['pop_size'], 
                mut_rate=config['mut_rate'], 
                generations=config['generations'],
                games=config['games']
            )
            
            # Save model
            model_data = {
                'DNA': best_model.DNA,
                'fitness': best_fitness,
                'pop_size': config['pop_size'],
                'mut_rate': config['mut_rate'],
                'score': best_score,
                'steps': best_steps,
                'generation': best_gen,
                'games': config['games']
            }
            
            model_filename = f"model_{i+1}_pop{config['pop_size']}_mut{config['mut_rate']}.pkl"
            model_path = os.path.join(self.ai_models_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            print(f"Model saved to {model_path}")
            print(f"Best fitness: {best_fitness:.2f}, Score: {best_score:.2f}, Steps: {best_steps:.2f}")
            
        print("\nAll models trained and saved successfully!")
    
    def select_best_model(self):
        """
        Evaluate all trained models and select the best one for the final experiment.
        
        Returns:
            str: Path to the best model file
        """
        from ga_snake import run_best_model
        
        print("Evaluating all trained models...")
        model_files = [f for f in os.listdir(self.ai_models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            print("No models found. Please run prepare_ga_models first.")
            return None
            
        results = []
        
        for model_file in model_files:
            model_path = os.path.join(self.ai_models_dir, model_file)
            print(f"\nEvaluating model: {model_file}")
            
            # Load model data to get parameters
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Run evaluation (without display)
            avg_score, avg_steps, _ = self._evaluate_model(model_path, 
                                                         num_games=10,
                                                         display=False)
            
            results.append({
                'model_file': model_file,
                'model_path': model_path,
                'pop_size': model_data['pop_size'],
                'mut_rate': model_data['mut_rate'],
                'generation': model_data['generation'],
                'validation_score': avg_score,
                'validation_steps': avg_steps
            })
            
        # Find the best model by validation score
        best_model = max(results, key=lambda x: x['validation_score'])
        
        # Save evaluation results
        eval_results_path = os.path.join(self.results_dir, 'model_evaluation_results.json')
        with open(eval_results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print("\nModel Evaluation Complete")
        print(f"Best Model: {best_model['model_file']}")
        print(f"Average Score: {best_model['validation_score']:.2f}")
        
        # Copy the best model to a standard location
        import shutil
        best_model_path = os.path.join(self.experiment_dir, 'best_model.pkl')
        shutil.copy(best_model['model_path'], best_model_path)
        
        print(f"Best model copied to {best_model_path}")
        return best_model_path
    
    def _evaluate_model(self, model_path, num_games=10, display=False):
        """
        Helper method to evaluate a model's performance.
        
        Returns:
            tuple: (avg_score, avg_steps, detailed_results)
        """
        from snake import SnakeGame
        from ga_models.ga_simple import SimpleModel
        
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Determine input dimension from model's DNA
        if 'DNA' in model_data and model_data['DNA']:
            input_dim = model_data['DNA'][0].shape[0]
        else:
            input_dim = 21  # Default
            
        print(f"Model input dimension: {input_dim}")
            
        # Create model with the correct dimensions
        if input_dim == 15:
            model = SimpleModel(dims=(15, 50, 50, 4))
        elif input_dim == 18:
            model = SimpleModel(dims=(18, 50, 50, 4))
        else:
            model = SimpleModel(dims=(21, 50, 50, 4))
            
        model.DNA = model_data['DNA']
        
        # Run games
        game = SnakeGame(xsize=self.config['game_parameters']['grid_size'], 
                         ysize=self.config['game_parameters']['grid_size'])
        scores = []
        steps_list = []
        
        for i in range(num_games):
            score, steps = game.run(
                model=model, 
                max_steps=self.config['game_parameters']['max_steps'], 
                display=display
            )
            scores.append(score)
            steps_list.append(steps)
            
        avg_score = np.mean(scores)
        avg_steps = np.mean(steps_list)
        
        detailed_results = {
            'scores': scores,
            'steps': steps_list,
            'avg_score': float(avg_score),
            'avg_steps': float(avg_steps),
            'max_score': int(np.max(scores))
        }
        
        return avg_score, avg_steps, detailed_results
    
    def collect_human_data(self):
        """
        Launch the human data collection tool and monitor progress.
        """
        from human_performance import HumanPerformanceTracker
        
        # Check if we already have enough data
        tracker = HumanPerformanceTracker()
        
        # Count total games played by humans
        total_games = sum(len(session['scores']) for session in tracker.results['sessions'])
        
        if total_games >= self.config['human_games_required']:
            print(f"Human data collection complete! ({total_games} games recorded)")
            return
            
        print(f"Human data collection in progress: {total_games}/{self.config['human_games_required']} games")
        print("Please run the human_performance.py script to collect more data.")
        
    def run_final_experiment(self):
        """
        Run the final comparison between the best AI model and human performance.
        """
        from compare_performance import PerformanceAnalyzer
        
        print("\n=== RUNNING FINAL EXPERIMENT ===\n")
        
        # Best model path
        best_model_path = os.path.join(self.experiment_dir, 'best_model.pkl')
        if not os.path.exists(best_model_path):
            best_model_path = self.select_best_model()
            if not best_model_path:
                return
                
        # Run performance analysis
        analyzer = PerformanceAnalyzer(ai_model_file=best_model_path)
        report = analyzer.generate_report()
        
        if report:
            # Save detailed report
            report_path = os.path.join(self.results_dir, 'final_comparison_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            # Generate additional visualizations
            self._generate_advanced_visualizations(report)
            
    def _generate_advanced_visualizations(self, report):
        """
        Generate additional visualizations for deeper analysis.
        
        Args:
            report: Performance comparison report data
        """
        # Create output directory for visualizations
        viz_dir = os.path.join(self.results_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
            
        # 1. Experience level vs. performance (boxplot)
        from human_performance import HumanPerformanceTracker
        tracker = HumanPerformanceTracker()
        
        player_data = []
        for session in tracker.results['sessions']:
            player_id = session['player_id']
            experience = tracker.results['players'][player_id]['experience_level']
            for score in session['scores']:
                player_data.append({
                    'player_id': player_id,
                    'experience': experience,
                    'score': score
                })
                
        if player_data:
            df = pd.DataFrame(player_data)
            
            plt.figure(figsize=(10, 6))
            df_grouped = df.groupby('experience')
            experience_levels = sorted(df['experience'].unique())
            
            boxplot_data = [df_grouped.get_group(exp)['score'] for exp in experience_levels]
            plt.boxplot(boxplot_data, labels=[f"Level {exp}" for exp in experience_levels])
            
            # Add AI performance reference line
            ai_avg_score = report['ai_performance']['avg_score']
            plt.axhline(y=ai_avg_score, color='r', linestyle='--', 
                       label=f'AI Avg Score ({ai_avg_score:.2f})')
            
            plt.xlabel('Player Experience Level')
            plt.ylabel('Score')
            plt.title('Score Distribution by Experience Level')
            plt.legend()
            
            plt.savefig(os.path.join(viz_dir, 'experience_vs_performance.png'))
            
        # 2. Learning curve for humans (experience vs. time)
        sessions_by_player = {}
        for session in tracker.results['sessions']:
            player_id = session['player_id']
            if player_id not in sessions_by_player:
                sessions_by_player[player_id] = []
            sessions_by_player[player_id].append(session)
            
        # Sort sessions by timestamp for each player
        for player_id, sessions in sessions_by_player.items():
            sessions_by_player[player_id] = sorted(sessions, key=lambda x: x['timestamp'])
            
        # Plot learning curves for players with multiple sessions
        plt.figure(figsize=(12, 8))
        for player_id, sessions in sessions_by_player.items():
            if len(sessions) > 1:
                player_name = tracker.results['players'][player_id]['name']
                avg_scores = [np.mean(session['scores']) for session in sessions]
                plt.plot(range(1, len(avg_scores) + 1), avg_scores, 
                        marker='o', label=f"{player_name}")
                
        plt.xlabel('Session Number')
        plt.ylabel('Average Score')
        plt.title('Human Learning Curves')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.savefig(os.path.join(viz_dir, 'human_learning_curves.png'))
            
        print(f"Advanced visualizations saved to {viz_dir}")
    

def main():
    """
    Main function to execute the experimental workflow.
    """
    print("Snake AI vs. Human Experimental Design Manager")
    print("---------------------------------------------")
    
    manager = ExperimentManager()
    
    while True:
        print("\nExperimental Workflow:")
        print("1. Setup experiment (create directory structure)")
        print("2. Prepare GA models (train multiple models)")
        print("3. Select best model (evaluate and select)")
        print("4. Collect human data (launch data collection tool)")
        print("5. Run final experiment (compare AI vs. human)")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ")
        
        if choice == '1':
            manager.setup_experiment()
            
        elif choice == '2':
            try:
                models = int(input("Number of models to train (default: 3): ") or "3")
                
                # Ask about advanced training
                advanced = input("Use advanced training? This takes longer but produces better models. (y/n): ").lower()
                advanced_training = advanced.startswith('y')
                
                manager.prepare_ga_models(models_to_evaluate=models, advanced_training=advanced_training)
            except ValueError:
                print("Invalid input. Using default values.")
                manager.prepare_ga_models()
                
        elif choice == '3':
            manager.select_best_model()
            
        elif choice == '4':
            manager.collect_human_data()
            
        elif choice == '5':
            manager.run_final_experiment()
            
        elif choice == '6':
            print("Exiting Experiment Manager.")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
