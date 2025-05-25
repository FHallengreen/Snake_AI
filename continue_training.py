import numpy as np
import pickle
import argparse
import time
import multiprocessing
import os
from functools import partial
from snake import SnakeGame
from ga_models.ga_simple import SimpleModel
from ga_train import evaluate_individual  # Reuse the fitness function

def create_population_from_model(base_model, pop_size, mutation_strength=0.1):
    """
    Create a population centered around a base model with various mutation levels.
    This is a specialized form of GA Initialization from an existing model.
    """
    population = [base_model]  # Include the original model
    
    # Create variations of the base model with different mutation strengths
    for i in range(pop_size - 1):
        # Clone the base model
        clone = SimpleModel(dims=base_model.dims)
        clone.DNA = [layer.copy() for layer in base_model.DNA]
        
        # Apply stronger mutations for initial diversity
        mutation_rate = mutation_strength * (1 + 0.5 * (i / pop_size))
        clone.mutate(mutation_rate)
        population.append(clone)
    
    return population

def continue_training(model_file='experiment_data/best_ai/best_model.pkl', 
                     pop_size=150, mut_rate=0.03, generations=50,
                     games=5, output_file='experiment_data/best_ai/enhanced_model.pkl'):
    """
    Continue training from an existing model with optimized parameters.
    
    This function follows the same GA structure as run_ga_experiment but starts
    from a pre-trained model:
    
    1. INITIALIZATION: Load model and create population around it
    2. EVALUATION: Calculate fitness for each individual
    3. SELECTION: Tournament selection and elitism
    4. CROSSOVER & MUTATION: Create offspring with genetic operators
    5. REPLACEMENT: Form new generation
    6. TERMINATION: Stop after max generations or when progress plateaus
    """
    print("=" * 60)
    print(f"CONTINUING TRAINING FROM {model_file}")
    print("=" * 60)
    
    # Load the existing model
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {model_file} not found.")
        return
    
    # Create a model from the saved DNA
    dna = model_data.get('DNA', None)
    if isinstance(dna, list) and len(dna) > 0:
        # Figure out the architecture from DNA shapes
        dims = [dna[0].shape[0]]
        for layer in dna:
            dims.append(layer.shape[1])
        
        print(f"Loaded model with architecture: {dims}")
        print(f"Previous training score: {model_data.get('score', 'unknown')}")
        
        base_model = SimpleModel(dims=tuple(dims))
        base_model.DNA = dna
        
        # GA STEP 1: INITIALIZATION - Load existing model and create population
        # Create population with advanced initialization
        print(f"Creating population of size {pop_size} based on loaded model...")
        population = create_population_from_model(base_model, pop_size, mutation_strength=0.12)
    else:
        print("Error: Could not extract valid DNA from the model file.")
        return
    
    # Initialize tracking variables
    stats = {'avg': [], 'max': [], 'avg_score': [], 'avg_steps': []}
    best_model = base_model
    best_fitness = -float('inf')
    best_score = 0
    best_steps = 0
    best_gen = 0
    
    # Modified early stopping parameters
    patience = 40  # Increased patience for continued training
    min_delta = 0.001  # More sensitive to small improvements
    stagnation_counter = 0
    last_best_fitness = -float('inf')
    
    # Set up parallel processing
    num_processes = min(multiprocessing.cpu_count(), pop_size)
    print(f"Using {num_processes} processes for parallel evaluation")
    
    # Training start time
    start_time = time.time()
    
    # GA MAIN LOOP
    for gen in range(generations):
        # GA STEP 2: FITNESS EVALUATION
        # Evaluate the population
        with multiprocessing.Pool(processes=num_processes) as pool:
            eval_func = partial(evaluate_individual, games=games)
            results = pool.map(eval_func, population)
        
        fitnesses, scores, steps = zip(*results)
        
        # Update stats
        stats['avg'].append(np.mean(fitnesses))
        stats['max'].append(np.max(fitnesses))
        stats['avg_score'].append(np.mean(scores) / games)
        stats['avg_steps'].append(np.mean(steps) / games)
        max_fitness = np.max(fitnesses)
        
        # Check if we have a new best model
        if max_fitness > best_fitness:
            improvement = max_fitness - last_best_fitness
            if improvement > min_delta:
                stagnation_counter = 0
                last_best_fitness = max_fitness
            
            best_fitness = max_fitness
            best_idx = np.argmax(fitnesses)
            best_model = population[best_idx]
            best_score = scores[best_idx] / games
            best_steps = steps[best_idx] / games
            best_gen = gen
            
            # Save intermediate best model every 10 generations if it's an improvement
            if gen % 10 == 0:
                intermediate_model_data = {
                    'DNA': best_model.DNA,
                    'fitness': best_fitness,
                    'score': best_score,
                    'steps': best_steps,
                    'generation': best_gen,
                    'games': games,
                    'source_model': model_file
                }
                # Get directory from output_file path
                output_dir = os.path.dirname(output_file)
                os.makedirs(output_dir, exist_ok=True)
                with open(f"{output_dir}/intermediate_{os.path.basename(output_file)}", 'wb') as f:
                    pickle.dump(intermediate_model_data, f)
        else:
            stagnation_counter += 1
            
        print(f"Generation {gen + 1}: Avg Score = {stats['avg_score'][-1]:.2f}, Max Score = {best_score:.2f}")
        
        # GA STEP 6: TERMINATION - Check for early stopping
        # Check for early stopping
        if stagnation_counter >= patience:
            print(f"Early stopping: No improvement for {patience} generations")
            break
        
        # Dynamic mutation rates - adjusted based on empirical results
        if gen < generations * 0.15:  # First 15%
            current_mut_rate = mut_rate * 0.9  # Start higher for continued training
        elif gen > generations * 0.7:  # Last 30%
            current_mut_rate = mut_rate * 0.35  # Very fine tuning
        else:  # Middle 55%
            current_mut_rate = mut_rate * 0.6  # Balanced refinement
        
        # GA STEP 3: SELECTION with ELITISM
        # Adaptive elitism - keep more of the good models as training progresses
        elite_percent = 0.2 + min(0.3, gen / generations * 0.2)  # 20% to 40% (higher than normal)
        elite_count = max(1, int(pop_size * elite_percent))
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        elites = [population[i] for i in elite_indices]
        
        # Create offspring
        offspring = []
        tournament_size = 5  # Increased tournament size for more selection pressure
        
        for _ in range(pop_size - elite_count):
            # Tournament selection
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            parent1 = population[tournament_indices[np.argmax(tournament_fitnesses)]]
            
            # Second parent via tournament
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            parent2 = population[tournament_indices[np.argmax(tournament_fitnesses)]]
            
            child = parent1 + parent2
            child.mutate(current_mut_rate)
            offspring.append(child)
        
        # Form new population
        population = elites + offspring
    
    # Save final best model
    final_model_data = {
        'DNA': best_model.DNA,
        'fitness': best_fitness,
        'score': best_score,
        'steps': best_steps,
        'generation': best_gen,
        'games': games,
        'continued_from': model_file
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(final_model_data, f)
    
    print("\n" + "=" * 60)
    print("CONTINUED TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model saved to {output_file}")
    print(f"Final score: {best_score:.2f} (Original: {model_data.get('score', 'unknown')})")
    print(f"Steps: {best_steps:.2f}")
    print(f"Fitness: {best_fitness:.2f}")
    print(f"Best generation: {best_gen + 1}")
    
    # Compare with original model
    original_score = model_data.get('score', 0)
    if original_score > 0:
        improvement = best_score - original_score
        percent_improvement = (improvement / original_score) * 100
        print(f"Improvement: {improvement:.2f} points ({percent_improvement:.1f}%)")
    
    return best_model, best_score, best_fitness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue training from an existing model")
    parser.add_argument('--model', type=str, default='experiment_data/best_ai/best_model.pkl',
                      help="Path to the model file to continue training from")
    parser.add_argument('--pop-size', type=int, default=150,
                      help="Population size for continued training")
    parser.add_argument('--mut-rate', type=float, default=0.033,
                      help="Base mutation rate")
    parser.add_argument('--generations', type=int, default=170,
                      help="Maximum number of generations to run")
    parser.add_argument('--games', type=int, default=7, 
                      help="Number of games to evaluate each individual")
    parser.add_argument('--output', type=str, default='experiment_data/best_ai/enhanced_model.pkl',
                      help="Output file for the enhanced model")
    
    args = parser.parse_args()
    
    continue_training(
        model_file=args.model,
        pop_size=args.pop_size,
        mut_rate=args.mut_rate,
        generations=args.generations,
        games=args.games,
        output_file=args.output
    )
