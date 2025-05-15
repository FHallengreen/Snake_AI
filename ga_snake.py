import pickle
import numpy as np
import argparse
import time
import os
from snake import SnakeGame
from ga_controller import GAController
from ga_models.ga_simple import SimpleModel


def detect_network_architecture(model_data):
    """
    Detect the network architecture from the model data.
    Returns a tuple of dimensions (input_dim, hidden_dims..., output_dim)
    """
    dna = model_data.get('DNA', None)
    if dna is None:
        return (21, 128, 64, 32, 4)  # Default architecture
    
    if isinstance(dna, list) and len(dna) > 0:
        # Extract dimensions from the weights
        dims = [dna[0].shape[0]]  # Input dimension
        for layer in dna:
            dims.append(layer.shape[1])  # Output dimension of each layer
        return tuple(dims)
    
    # Fallback to default
    return (21, 128, 64, 32, 4)


def run_best_model(model_file='experiment_data/best_ai/enhanced_model_final.pkl', 
                   num_games=10, max_steps=2000, display=False,
                   save_stats=False, output_dir=None):
    """
    Run and evaluate a trained model.
    
    Args:
        model_file: Path to the model file (.pkl)
        num_games: Number of games to run for evaluation
        max_steps: Maximum steps per game
        display: Whether to display the game visually
        save_stats: Whether to save the evaluation statistics
        output_dir: Directory to save statistics to (if save_stats is True)
    """
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {model_file} not found.")
        return
    
    # Detect the network architecture
    dims = detect_network_architecture(model_data)
    
    print(f"Model architecture: {dims}")
    model = SimpleModel(dims=dims)
    model.DNA = model_data['DNA']
    
    # Display model information
    print(f"\nLoaded model from {model_file}")
    print(f"Fitness: {model_data.get('fitness', 'Unknown'):.2f}")
    if 'pop_size' in model_data:
        print(f"Training parameters: Pop Size={model_data['pop_size']}, Mut Rate={model_data['mut_rate']:.4f}")
    print(f"Training Score: {model_data.get('score', 'Unknown'):.2f}, Steps: {model_data.get('steps', 'Unknown'):.2f}")
    
    # Check if this is a continued model
    if 'continued_from' in model_data:
        print(f"This model was continued from: {model_data['continued_from']}")
        
    # Run evaluation games
    game = SnakeGame(xsize=20, ysize=20)
    scores = []
    steps_list = []
    start_time = time.time()
    
    print(f"\nRunning {num_games} evaluation games {'with display' if display else 'silently'}...")
    
    for i in range(num_games):
        score, steps = game.run(model=model, max_steps=max_steps, display=display)
        scores.append(score)
        steps_list.append(steps)
        print(f"Game {i + 1}: Score = {score}, Steps = {steps}")
        
        # Add small delay between displayed games to make them easier to follow
        if display:
            time.sleep(1)
    
    elapsed = time.time() - start_time
    
    # Calculate statistics
    avg_score = np.mean(scores)
    med_score = np.median(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    avg_steps = np.mean(steps_list)
    
    print(f"\n{'=' * 30}")
    print(f"Performance Summary ({num_games} games):")
    print(f"{'=' * 30}")
    print(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
    print(f"Median Score: {med_score:.2f}")
    print(f"Max Score: {max_score:.2f}")
    print(f"Min Score: {min_score:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Total Evaluation Time: {elapsed:.2f} seconds")
    
    # Save statistics if requested
    if save_stats:
        stats = {
            'model_file': model_file,
            'num_games': num_games,
            'max_steps': max_steps,
            'avg_score': float(avg_score),
            'med_score': float(med_score),
            'std_score': float(std_score),
            'max_score': float(max_score),
            'min_score': float(min_score),
            'avg_steps': float(avg_steps),
            'evaluation_time': float(elapsed),
            'scores': [float(s) for s in scores],
            'steps': [float(s) for s in steps_list]
        }
        
        # Create default output directory if not specified
        if output_dir is None:
            output_dir = 'evaluation_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename based on model name
        model_name = os.path.basename(model_file).split('.')[0]
        out_file = f"{output_dir}/{model_name}_eval_{num_games}games.pkl"
        
        with open(out_file, 'wb') as f:
            pickle.dump(stats, f)
        print(f"\nStatistics saved to {out_file}")
    
    return avg_score, scores


def compare_models(original='experiment_data/best_model.pkl', 
                  enhanced='enhanced_model.pkl',
                  num_games=20):
    """Compare the original model with an enhanced version."""
    print("\n" + "=" * 50)
    print("COMPARING MODELS")
    print("=" * 50)
    
    print("\nEvaluating original model:")
    orig_avg, orig_scores = run_best_model(model_file=original, num_games=num_games)
    
    print("\nEvaluating enhanced model:")
    enhanced_avg, enhanced_scores = run_best_model(model_file=enhanced, num_games=num_games)
    
    # Calculate improvement
    improvement = enhanced_avg - orig_avg
    percent_improvement = (improvement / orig_avg) * 100 if orig_avg > 0 else float('inf')
    
    # Statistical comparison
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"Original model average score: {orig_avg:.2f}")
    print(f"Enhanced model average score: {enhanced_avg:.2f}")
    print(f"Absolute improvement: {improvement:.2f} points")
    print(f"Relative improvement: {percent_improvement:.1f}%")
    
    # Perform statistical test if we have enough samples
    if num_games >= 10:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(enhanced_scores, orig_scores)
        print(f"\nStatistical significance: p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("The improvement is statistically significant (p < 0.05)")
        else:
            print("The improvement is not statistically significant (p >= 0.05)")


def run_test_evaluation(model_file='experiment_data/best_ai/enhanced_model_final.pkl', 
                      num_games=30, custom_config=False):
    """
    Run a formal test evaluation.
    
    Args:
        model_file: Path to the model file to test
        num_games: Number of test games to run (should be 30+ for statistical validity)
        custom_config: Unused parameter (kept for backward compatibility)
    """
    print("\n" + "=" * 60)
    print("FORMAL TEST EVALUATION")
    print("=" * 60)
    
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {model_file} not found.")
        return
    
    # Load the model
    dims = detect_network_architecture(model_data)
    model = SimpleModel(dims=dims)
    model.DNA = model_data['DNA']
    
    # Only use the standard configuration - no custom board sizes
    test_configs = [
        {"xsize": 20, "ysize": 20, "max_steps": 2000},  # Standard configuration
    ]
    
    all_scores = []
    all_steps = []
    config_results = []
    
    # Run tests for each configuration
    for config_idx, config in enumerate(test_configs):
        print(f"\nTest configuration {config_idx+1}: Grid size {config['xsize']}×{config['ysize']}")
        
        game = SnakeGame(xsize=config['xsize'], ysize=config['ysize'])
        scores = []
        steps_list = []
        
        for i in range(num_games):
            score, steps = game.run(model=model, max_steps=config['max_steps'])
            scores.append(score)
            steps_list.append(steps)
            
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
        print(f"Average Steps: {np.mean(steps_list):.2f}")
        
        all_scores.extend(scores)
        all_steps.extend(steps_list)
        config_results.append({
            "config": config,
            "avg_score": float(avg_score),
            "std_score": float(std_score),
            "avg_steps": float(np.mean(steps_list)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores))
        })
    
    # Overall results
    print("\n" + "=" * 60)
    print(f"OVERALL TEST RESULTS ({len(all_scores)} games)")
    print("=" * 60)
    print(f"Average Score: {np.mean(all_scores):.2f} ± {np.std(all_scores):.2f}")
    print(f"Median Score: {np.median(all_scores):.2f}")
    print(f"Average Steps: {np.mean(all_steps):.2f}")
    
    # Save test results
    test_results = {
        "model_file": model_file,
        "num_games": num_games,
        "overall_avg_score": float(np.mean(all_scores)),
        "overall_std_score": float(np.std(all_scores)),
        "overall_median_score": float(np.median(all_scores)),
        "config_results": config_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs("test_results", exist_ok=True)
    model_name = os.path.basename(model_file).split('.')[0]
    result_file = f"test_results/{model_name}_test_{num_games}games.json"
    
    import json
    with open(result_file, 'w') as f:
        json.dump(test_results, f, indent=4)
    
    print(f"\nTest results saved to {result_file}")
    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation of trained Snake AI models")
    parser.add_argument('--model', type=str, default='experiment_data/best_ai/enhanced_model_final.pkl',
                      help="Path to the model file to evaluate")
    parser.add_argument('--games', type=int, default=10, 
                      help="Number of games to run for evaluation")
    parser.add_argument('--steps', type=int, default=2000,
                      help="Maximum steps per game")
    parser.add_argument('--display', action='store_true', 
                      help="Display the game visually")
    parser.add_argument('--save-stats', action='store_true',
                      help="Save evaluation statistics to a file")
    parser.add_argument('--compare', type=str, default=None,
                      help="Path to another model to compare with")
    parser.add_argument('--test', action='store_true',
                      help="Run formal test evaluation")
    parser.add_argument('--test-games', type=int, default=30,
                      help="Number of games for formal testing (default: 30)")
    
    args = parser.parse_args()
    
    if args.test:
        # Run formal test evaluation
        run_test_evaluation(
            model_file=args.model,
            num_games=args.test_games
        )
    elif args.compare:
        # Compare this model with another
        compare_models(original=args.model, enhanced=args.compare, num_games=args.games)
    else:
        # Run single model evaluation
        run_best_model(
            model_file=args.model,
            num_games=args.games,
            max_steps=args.steps,
            display=args.display,
            save_stats=args.save_stats
        )