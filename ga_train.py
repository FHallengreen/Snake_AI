import numpy as np
import csv
import pickle
from snake import SnakeGame
from ga_models.ga_simple import SimpleModel


def evaluate_individual(model, games=3):
    game = SnakeGame(xsize=20, ysize=20)
    total_score = 0
    total_steps = 0
    collisions = 0
    for _ in range(games):
        score, steps = game.run(model=model, max_steps=2000, display=False)
        total_score += score
        total_steps += steps
        if not game.snake.p.within(game.grid) or game.snake.cross_own_tail:
            collisions += 1
        # Calculate food proximity (simplified for evaluation)
        dfx = game.snake.p.x - game.food.p.x
        dfy = game.snake.p.y - game.food.p.y
        dist_to_food = abs(dfx) + abs(dfy)
        food_proximity = 100 * (1 - dist_to_food / (game.grid.x + game.grid.y))
    fitness = total_score * 10000 + total_steps + food_proximity - collisions * 100
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
    population = [SimpleModel(dims=(15, 50, 50, 4)) for _ in range(pop_size)]
    stats = {'avg': [], 'max': [], 'avg_score': [], 'avg_steps': []}
    best_model = None
    best_fitness = -float('inf')
    best_score = 0
    best_steps = 0
    best_gen = 0
    for gen in range(generations):
        results = [evaluate_individual(ind, games=games) for ind in population]
        fitnesses, scores, steps = zip(*results)
        stats['avg'].append(np.mean(fitnesses))
        stats['max'].append(np.max(fitnesses))
        stats['avg_score'].append(np.mean(scores) / games)
        stats['avg_steps'].append(np.mean(steps) / games)
        max_fitness = np.max(fitnesses)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_idx = np.argmax(fitnesses)
            best_model = population[best_idx]
            best_score = scores[best_idx] / games
            best_steps = steps[best_idx] / games
            best_gen = gen
        print(f"Generation {gen + 1}: Avg Score = {stats['avg_score'][-1]:.2f}, Max Score = {best_score:.2f}")
        # Elitism: Keep top 10%
        elite_count = max(1, pop_size // 10)
        elites = [population[i] for i in np.argsort(fitnesses)[-elite_count:]]
        offspring = []
        for _ in range(pop_size - elite_count):
            parent1 = roulette_wheel_selection(population, fitnesses)
            parent2 = roulette_wheel_selection(population, fitnesses)
            child = parent1 + parent2
            child.mutate(mut_rate)
            offspring.append(child)
        population = elites + offspring
    return stats, best_model, best_fitness, best_score, best_steps, best_gen


if __name__ == "__main__":
    test_mode = False
    if test_mode:
        params = [(50, 0.2)]  # Increased mutation rate
        generations = 50
        games = 1
    else:
        params = [
            (50, 0.01), (50, 0.05), (50, 0.1),
            (100, 0.01), (100, 0.05), (100, 0.1),
            # (200, 0.01), (200, 0.05), (200, 0.1)
        ]
        generations = 100
        games = 3
    results = []
    best_overall_model = None
    best_overall_fitness = -float('inf')
    best_overall_params = None
    best_overall_score = 0
    best_overall_steps = 0
    best_overall_gen = 0
    for pop_size, mut_rate in params:
        print(f"Running GA with pop_size={pop_size}, mut_rate={mut_rate}")
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
                'pop_size': best_overall_params[0],
                'mut_rate': best_overall_params[1],
                'score': best_overall_score,
                'steps': best_overall_steps,
                'generation': best_overall_gen,
                'games': games
            }, f)
        print("\nBest Model Saved:")
        print(f"Pop Size: {best_overall_params[0]}, Mut Rate: {best_overall_params[1]}")
        print(f"Fitness: {best_overall_fitness:.2f} (Score: {best_overall_score:.2f}, Steps: {best_overall_steps:.2f})")
        print(f"Generation: {best_overall_gen + 1}")