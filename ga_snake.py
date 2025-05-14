#!/usr/bin/env python
import pickle
import numpy as np
from snake import SnakeGame
from ga_controller import GAController
from ga_models.ga_simple import SimpleModel


def run_best_model(model_file='best_model.pkl', num_games=5, max_steps=2000):
    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print("Error: best_model.pkl not found. Run ga_train.py first.")
        return
    model = SimpleModel(dims=(15, 50, 50, 4))
    model.DNA = model_data['DNA']
    print(f"Loaded model with fitness {model_data['fitness']:.2f} "
          f"(Pop Size: {model_data['pop_size']}, Mut Rate: {model_data['mut_rate']}, "
          f"Generation: {model_data['generation'] + 1})")
    print(f"Training Score: {model_data['score']:.2f}, "
          f"Steps: {model_data['steps']:.2f}")
    game = SnakeGame(xsize=20, ysize=20)
    scores = []
    steps_list = []
    for i in range(num_games):
        score, steps = game.run(model=model, max_steps=max_steps, display=True)
        scores.append(score)
        steps_list.append(steps)
        print(f"Game {i + 1}: Score = {score}, Steps = {steps}")
    print(f"\nAverage Performance (over {num_games} games):")
    print(f"Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Steps: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}")


if __name__ == "__main__":
    run_best_model(num_games=10, max_steps=2000)