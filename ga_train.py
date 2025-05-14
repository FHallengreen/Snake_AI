import numpy as np
import csv
import pickle
import multiprocessing
from functools import partial
from snake import SnakeGame
from ga_models.ga_simple import SimpleModel


def evaluate_individual(model, games=3):
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
    
    # Significantly improved fitness function:
    # 1. Major reward for food eaten (primary goal)
    # 2. Reward for food proximity and progress toward food
    # 3. Survival time reward but with diminishing returns
    # 4. Severe penalties for collisions and not eating
    fitness = (
        # Primary goal: eating food
        (food_eaten * 20000) + 
        
        # Getting closer to food
        (max_distance_reduced * 500) + 
        (food_rewards * 50) +
        
        # Survival time with diminishing returns
        (min(total_steps, 1000) * 2) +  # Full value for first 1000 steps
        (max(0, total_steps - 1000) * 0.5) +  # Half value for steps beyond 1000
        
        # Penalties
        (collisions * -2000) -  # Severe penalty for collisions
        (total_steps_without_food * 10)  # Increased penalty for not eating
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


def run_ga_experiment(pop_size, mut_rate, generations=100, games=3):
    # Use a deeper, wider network architecture with a new layer structure
    population = [SimpleModel(dims=(21, 128, 64, 32, 4)) for _ in range(pop_size)]
    stats = {'avg': [], 'max': [], 'avg_score': [], 'avg_steps': []}
    best_model = None
    best_fitness = -float('inf')
    best_score = 0
    best_steps = 0
    best_gen = 0
    
    # Early stopping parameters
    patience = 30  # Number of generations with no improvement to wait before stopping
    min_delta = 0.01  # Minimum change in fitness to qualify as improvement
    stagnation_counter = 0
    last_best_fitness = -float('inf')
    
    # Determine optimal number of processes based on CPU cores
    num_processes = min(multiprocessing.cpu_count(), pop_size)
    print(f"Using {num_processes} processes for parallel evaluation")
    
    for gen in range(generations):
        # Parallelize the evaluation of individuals
        with multiprocessing.Pool(processes=num_processes) as pool:
            eval_func = partial(evaluate_individual, games=games)
            results = pool.map(eval_func, population)
        
        fitnesses, scores, steps = zip(*results)
        stats['avg'].append(np.mean(fitnesses))
        stats['max'].append(np.max(fitnesses))
        stats['avg_score'].append(np.mean(scores) / games)
        stats['avg_steps'].append(np.mean(steps) / games)
        max_fitness = np.max(fitnesses)
        
        # Check if we have a new best model with meaningful improvement
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
        
        # Early stopping check - if no improvement for 'patience' generations, stop training
        if stagnation_counter >= patience:
            print(f"Early stopping at generation {gen + 1}: No significant improvement for {patience} generations")
            break
        
        # Improved selection and reproduction strategy
        # Elitism: Keep top 15% (increased from 10%)
        elite_count = max(1, pop_size * 15 // 100)
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        elites = [population[i] for i in elite_indices]
        
        # Use a more aggressive early-stage mutation rate that decreases over time
        # Based on observed learning curves where significant gains happen in early-mid training
        if gen < generations * 0.2:  # First 20% of generations - exploration phase
            current_mut_rate = mut_rate * 1.2
        elif gen > generations * 0.7:  # Last 30% of generations - fine tuning
            current_mut_rate = mut_rate * 0.5
        elif gen > generations * 0.5:  # Middle generations
            current_mut_rate = mut_rate * 0.75
        else:
            current_mut_rate = mut_rate
        
        offspring = []
        # Tournament selection with larger tournaments for better models
        tournament_size = 4  # Increase from 3 to 4
        
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
        
        population = elites + offspring
    
    return stats, best_model, best_fitness, best_score, best_steps, best_gen


if __name__ == "__main__":
    test_mode = False
    if test_mode:
        params = [(50, 0.3)]  # Higher mutation rate for testing
        generations = 50
        games = 1
    else:
        # Updated parameter configurations for shorter but effective training
        params = [
            # Population, mutation rate, generations, games per evaluation
            (150, 0.03, 200, 5),   # Medium population, balanced mutation
            (160, 0.035, 180, 5),  # Slightly higher mutation
            (170, 0.025, 180, 5),  # Slightly lower mutation
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
            generations = param_set[2] if len(param_set) > 2 else 200
            games = param_set[3] if len(param_set) > 3 else 3
            
            print(f"Running GA with pop_size={pop_size}, mut_rate={mut_rate}, generations={generations}, games={games}")
            stats, best_model, best_fitness, best_score, best_steps, best_gen = run_ga_experiment(
                pop_size, mut_rate, generations=generations, games=games
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
                      f"{best_score:9.2f} | {best_steps:9.2f}")
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