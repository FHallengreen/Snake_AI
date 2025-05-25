import numpy as np
import csv
import pickle
import multiprocessing
from functools import partial
import argparse
import time
from snake import SnakeGame
from ga_models.ga_simple import SimpleModel


def evaluate_individual(model, games=3):
    """
    Fitness function: Evaluate an individual model by playing multiple games
    and calculating a fitness score based on performance.
    
    This is the GA Fitness Evaluation component.
    """
    game = SnakeGame(xsize=20, ysize=20)
    total_score = 0
    total_steps = 0
    collisions = 0
    food_rewards = 0
    max_distance_reduced = 0
    total_steps_without_food = 0
    food_eaten = 0  # New metric to specifically track food consumption
    
    for game_num in range(games):
        # Track food-seeking metrics
        prev_distance = float('inf')
        steps_since_last_food = 0
        initial_dist = 0
        game.reset()
        
        # Calculate initial distance to food
        dfx = game.snake.p.x - game.food.p.x
        dfy = game.snake.p.y - game.food.p.y
        initial_dist = abs(dfx) + abs(dfy)
        
        # Track food eaten in this specific game
        starting_score = game.snake.score
        
        score, steps = game.run(model=model, max_steps=2000, display=False)
        total_score += score
        total_steps += steps
        
        # Increment food eaten count
        food_eaten += score
        
        # More aggressive penalty for not eating any food
        if score == 0:
            total_steps_without_food += steps
        
        # Penalty for collision
        if not game.snake.p.within(game.grid) or game.snake.cross_own_tail:
            collisions += 1
        
        # Track continuous progress toward food
        food_progress_rewards = 0
        min_dist_to_food = float('inf')
        for i in range(min(steps, 100)):  # Sample first 100 steps or fewer if game ended earlier
            dfx = abs(game.snake.p.x - game.food.p.x)
            dfy = abs(game.snake.p.y - game.food.p.y)
            current_dist = dfx + dfy
            
            # Update minimum distance
            if current_dist < min_dist_to_food:
                min_dist_to_food = current_dist
                food_progress_rewards += 5  # Reward for getting closer
        
        # Calculate final distance to food
        dfx = abs(game.snake.p.x - game.food.p.x)
        dfy = abs(game.snake.p.y - game.food.p.y)
        final_dist = dfx + dfy
        
        # Calculate how much the snake reduced the distance to food
        distance_reduced = initial_dist - final_dist
        if distance_reduced > 0:
            max_distance_reduced += distance_reduced
        
        # Calculate food proximity score - higher when closer to food
        food_proximity = 200 * (1 - final_dist / (game.grid.x + game.grid.y))
        food_rewards += food_proximity + food_progress_rewards
    
    # Normalized fitness function with better scaling
    fitness = (
        # Primary goal: eating food with progressive scaling
        (food_eaten * 100) * (1.0 + food_eaten * 0.1) + 
        
        # Getting closer to food - more reward for distance reduction 
        (max_distance_reduced * 3) + 
        (food_rewards * 0.3) +
        
        # Survival time with diminishing returns
        (min(total_steps, 1000) * 0.01) +  
        (max(0, total_steps - 1000) * 0.005) + 
        
        # Bonus for high scores with long survival
        (food_eaten > 20) * (total_steps * 0.03) +
        
        # Penalties
        (collisions * -20) - 
        (total_steps_without_food * 0.05)
    )
    
    return fitness, total_score, total_steps


def roulette_wheel_selection(population, fitnesses):
    min_fitness = min(fitnesses)
    shifted_fitnesses = [f + abs(min_fitness) if min_fitness < 0 else f for f in fitnesses]
    total_fitness = sum(shifted_fitnesses)
    if total_fitness == 0:
        return np.random.choice(population)
    probabilities = [f / total_fitness for f in shifted_fitnesses]
    return np.random.choice(population, p=probabilities)


def run_ga_experiment(pop_size, mut_rate, generations=100, games=3, network_arch=(21, 128, 64, 32, 4), model_class=SimpleModel):
    """
    Run a genetic algorithm experiment with the specified parameters.
    
    This function implements the main GA loop with:
    - Initialization: Create initial population
    - Evaluation: Calculate fitness for each individual  
    - Selection: Tournament selection to choose parents
    - Crossover & Mutation: Create offspring with genetic operators
    - Replacement: Form new generation with elitism
    - Termination: Stop after max generations or when progress plateaus
    
    Args:
        pop_size: Size of the population
        mut_rate: Base mutation rate
        generations: Maximum number of generations to run
        games: Number of games to evaluate each individual
        network_arch: Neural network architecture as a tuple of layer sizes
        model_class: The model class to use (SimpleModel or AdvancedModel)
    """
    # Initialize population with the specified network architecture and model class
    print(f"Using network architecture: {network_arch}")
    print(f"Model class: {model_class.__name__}")
    
    # GA STEP 1: INITIALIZATION - Create initial population
    if hasattr(model_class, '__name__') and model_class.__name__ == 'AdvancedModel':
        population = [model_class(dims=network_arch, use_residual=True) for _ in range(pop_size)]
    else:
        population = [model_class(dims=network_arch) for _ in range(pop_size)]
    
    # Statistics tracking
    stats = {'avg': [], 'max': [], 'avg_score': [], 'avg_steps': []}
    best_model = None
    best_fitness = -float('inf')
    best_score = 0
    best_steps = 0
    best_gen = 0
    
    # Early stopping parameters
    patience = 35
    min_delta = 0.005
    stagnation_counter = 0
    last_best_fitness = -float('inf')
    
    # Parallel processing setup
    num_processes = min(multiprocessing.cpu_count(), pop_size)
    print(f"Using {num_processes} processes for parallel evaluation")
    
    # Track time to estimate completion
    start_time = time.time()
    
    # GA MAIN LOOP - Iterate through generations
    for gen in range(generations):
        # GA STEP 2: FITNESS EVALUATION - Evaluate each individual
        with multiprocessing.Pool(processes=num_processes) as pool:
            eval_func = partial(evaluate_individual, games=games)
            results = pool.map(eval_func, population)
        
        fitnesses, scores, steps = zip(*results)
        stats['avg'].append(np.mean(fitnesses))
        stats['max'].append(np.max(fitnesses))
        stats['avg_score'].append(np.mean(scores) / games)
        stats['avg_steps'].append(np.mean(steps) / games)
        max_fitness = np.max(fitnesses)
        
        # Track best individual across generations
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
        else:
            stagnation_counter += 1
            
        print(f"Generation {gen + 1}: Avg Score = {stats['avg_score'][-1]:.2f}, Max Score = {best_score:.2f}")
        
        # GA STEP 6: TERMINATION - Check for early stopping
        if stagnation_counter >= patience:
            print(f"Early stopping at generation {gen + 1}: No significant improvement for {patience} generations")
            break
        
        # Dynamic mutation rate scheduling
        if gen < generations * 0.15:  # First 15% - high exploration
            current_mut_rate = mut_rate * 1.5
        elif gen < generations * 0.4:  # Early phase - moderate exploration
            current_mut_rate = mut_rate * 1.2
        elif gen > generations * 0.8:  # Late phase - fine tuning
            current_mut_rate = mut_rate * 0.4
        elif gen > generations * 0.6:  # Middle-late phase
            current_mut_rate = mut_rate * 0.6
        else:  # Middle phase
            current_mut_rate = mut_rate * 0.8
            
        # GA STEP 3: SELECTION with ELITISM - Preserve best individuals
        elite_percent = 0.15 + min(0.2, gen / generations * 0.15)
        elite_count = max(1, int(pop_size * elite_percent))
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        elites = [population[i] for i in elite_indices]
        
        # Create offspring
        offspring = []
        tournament_size = 4
        
        # Generate new individuals to maintain population size
        for _ in range(pop_size - elite_count):
            # GA STEP 3: SELECTION - Tournament selection for parents
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            parent1 = population[tournament_indices[np.argmax(tournament_fitnesses)]]
            
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            parent2 = population[tournament_indices[np.argmax(tournament_fitnesses)]]
            
            # GA STEP 4: CROSSOVER - Create child from parents
            child = parent1 + parent2
            
            # GA STEP 5: MUTATION - Apply mutation to child
            child.mutate(current_mut_rate)
            offspring.append(child)
        
        # Form new generation
        population = elites + offspring
    
    return stats, best_model, best_fitness, best_score, best_steps, best_gen


if __name__ == "__main__":
    # Add command line arguments for quick testing and configuration
    parser = argparse.ArgumentParser(description='Train snake AI using genetic algorithm')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with reduced settings')
    parser.add_argument('--network', type=str, default='medium', choices=['small', 'medium', 'large'],
                       help='Network size: small (21,64,32,4), medium (21,128,64,32,4), large (21,256,128,64,4)')
    args = parser.parse_args()
    
    if args.quick_test:
        print("Running quick test with reduced settings")
        params = [(100, 0.03, 50, 3)]  # Faster test run
    else:
        # Updated configurations based on previous results
        # The best model was from pop=170, mut=0.025
        # Let's try variations around this point
        network_arch = {
            'small': (21, 64, 32, 4),
            'medium': (21, 128, 64, 32, 4),
            'large': (21, 256, 128, 64, 4)
        }[args.network]
        
        params = [
            # Population, mutation rate, generations, games per evaluation
            (170, 0.025, 200, 5),  # Best configuration from previous run
            (180, 0.022, 180, 5),  # Slightly larger population, lower mutation
            (190, 0.020, 180, 5),  # Even larger population, even lower mutation
        ]
    
    results = []
    best_overall_model = None
    best_overall_fitness = -float('inf')
    best_overall_params = None
    best_overall_score = 0
    best_overall_steps = 0
    best_overall_gen = 0
    
    for param_set in params:
        pop_size, mut_rate = param_set[0], param_set[1]
        generations = param_set[2] if len(param_set) > 2 else 150  # Reduced default from 200
        games = param_set[3] if len(param_set) > 3 else 3
        
        print(f"Running GA with pop_size={pop_size}, mut_rate={mut_rate}, generations={generations}, games={games}")
        stats, best_model, best_fitness, best_score, best_steps, best_gen = run_ga_experiment(
            pop_size, mut_rate, generations=generations, games=games, network_arch=network_arch
        )
        results.append((pop_size, mut_rate, stats['avg'], stats['max'], stats['avg_score'], stats['avg_steps']))
        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_model = best_model
            best_overall_params = (pop_size, mut_rate)
            best_overall_score = best_score
            best_overall_steps = best_steps
            best_overall_gen = best_gen
    
    print("\nSummary of Results (Last 10 Generations):")
    print("Pop Size | Mut Rate | Avg Score | Avg Steps | Max Score | Max Steps")
    print("-" * 60)
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Pop Size', 'Mut Rate', 'Generation', 'Avg Fitness', 'Max Fitness', 'Avg Score', 'Avg Steps'])
        for pop_size, mut_rate, avg_fitness, max_fitness, avg_scores, avg_steps in results:
            last_10_scores = avg_scores[-10:] if len(avg_scores) >= 10 else avg_scores
            last_10_steps = avg_steps[-10:] if len(avg_steps) >= 10 else avg_steps
            print(f"{pop_size:8} | {mut_rate:8.2f} | {np.mean(last_10_scores):9.2f} | {np.mean(last_10_steps):9.2f} | "
                  f"{best_overall_score:9.2f} | {best_overall_steps:9.2f}")
            for gen, (avg_f, max_f, avg_s, avg_t) in enumerate(zip(avg_fitness, max_fitness, avg_scores, avg_steps)):
                writer.writerow([pop_size, mut_rate, gen + 1, avg_f, max_f, avg_s, avg_t])
    if best_overall_model:
        with open('best_model.pkl', 'wb') as f:
            pickle.dump({
                'DNA': best_overall_model.DNA,
                'fitness': best_overall_fitness,
                'score': best_overall_score,
                'steps': best_overall_steps,
                'pop_size': best_overall_params[0],
                'mut_rate': best_overall_params[1],
                'generation': best_overall_gen,
                'games': games
            }, f)
        print("\nBest Model Saved:")
        print(f"Pop Size: {best_overall_params[0]}, Mut Rate: {best_overall_params[1]}")
        print(f"Fitness: {best_overall_fitness:.2f} (Score: {best_overall_score:.2f}, Steps: {best_overall_steps:.2f})")
        print(f"Generation: {best_overall_gen + 1}")