#!/usr/bin/env python
"""
Module for tracking human player performance in the Snake game.
This allows collecting data from human play sessions to compare with AI performance.
"""
import json
import os
import time
import numpy as np
from snake import SnakeGame
from game_controller import HumanController

class HumanPerformanceTracker:
    def __init__(self, data_file='human_results.json'):
        self.data_file = data_file
        self.results = self._load_results()
    
    def _load_results(self):
        """Load existing results or create new results structure"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {self.data_file}, creating new results")
        return {'players': {}, 'sessions': []}
    
    def _save_results(self):
        """Save results to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def add_player(self, player_id, name, experience_level):
        """Add a new player to the tracking system
        
        Args:
            player_id: Unique identifier for the player
            name: Player's name
            experience_level: Self-reported experience (1-10)
        """
        if player_id not in self.results['players']:
            self.results['players'][player_id] = {
                'name': name,
                'experience_level': experience_level,
                'created': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self._save_results()
            print(f"Player {name} added successfully!")
        else:
            print(f"Player ID {player_id} already exists")
    def record_session(self, player_id, num_games=5, max_steps=2000, record_detailed_data=True):
        """Record a play session for a player
        
        Args:
            player_id: ID of the player
            num_games: Number of games to play in the session
            max_steps: Maximum steps per game
            record_detailed_data: Whether to record detailed gameplay data for strategy analysis
            
        Returns:
            dict: Session results summary
        """
        if player_id not in self.results['players']:
            print(f"Player ID {player_id} not found. Please add player first.")
            return None
        
        session_id = len(self.results['sessions'])
        player_name = self.results['players'][player_id]['name']
        
        print(f"\nStarting session for {player_name}")
        print(f"Play {num_games} games using arrow keys to control the snake.")
        print("Press any key to start...")
        input()
        
        game = SnakeGame(xsize=20, ysize=20)
        controller = HumanController(game)
        game.controller = controller
        
        scores = []
        steps_list = []
        detailed_data = [] if record_detailed_data else None
        
        for i in range(num_games):
            print(f"\nGame {i+1}/{num_games}")
            print("Use arrow keys to move. Eat food to grow. Don't hit walls or yourself!")
            game.reset()
            
            # Setup data collection if needed
            if record_detailed_data:
                moves = []
                food_positions = []
                snake_positions = []
                danger_states = []
                
                # Run the game with data recording
                score, steps = game.run(
                    max_steps=max_steps,
                    record_data=True,
                    moves=moves,
                    food_positions=food_positions,
                    snake_positions=snake_positions,
                    danger_states=danger_states
                )
                
                # Calculate additional metrics for strategy analysis
                food_efficiency = score / steps if steps > 0 else 0
                
                # Calculate average distance to food over time
                distances = []
                for j in range(min(len(snake_positions), len(food_positions))):
                    snake_head = snake_positions[j][0]  # First position is the head
                    food_pos = food_positions[j]
                    manhattan_dist = abs(snake_head[0] - food_pos[0]) + abs(snake_head[1] - food_pos[1])
                    distances.append(manhattan_dist)
                
                avg_distance = np.mean(distances) if distances else 0
                
                # Add game data to detailed data
                detailed_data.append({
                    'game_number': i + 1,
                    'score': score,
                    'steps': steps,
                    'food_efficiency': food_efficiency,
                    'avg_distance_to_food': avg_distance,
                    'moves': moves,
                    'food_positions': food_positions,
                    'snake_positions': snake_positions,
                    'danger_states': danger_states
                })
            else:
                # Run the game without data recording
                score, steps = game.run(max_steps=max_steps)
                
            scores.append(score)
            steps_list.append(steps)
            print(f"Game {i+1} finished: Score = {score}, Steps = {steps}")
            
            # Short pause between games
            if i < num_games - 1:
                print("Next game starting in 3 seconds...")
                time.sleep(3)
        
        # Calculate statistics
        avg_score = np.mean(scores)
        avg_steps = np.mean(steps_list)
        max_score = np.max(scores)
        
        # Record session data
        session_data = {
            'session_id': session_id,
            'player_id': player_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_games': num_games,
            'scores': scores,
            'steps': steps_list,
            'avg_score': float(avg_score),
            'avg_steps': float(avg_steps),
            'max_score': int(max_score)
        }
        
        # Add detailed data if collected
        if detailed_data:
            session_data['detailed_data'] = detailed_data
        
        self.results['sessions'].append(session_data)
        self._save_results()
        
        print("\nSession Summary:")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Maximum Score: {max_score}")
        
        return session_data
    
    def get_player_stats(self, player_id=None):
        """Get statistics for a specific player or all players
        
        Args:
            player_id: Optional ID of player to get stats for
            
        Returns:
            dict: Player statistics
        """
        player_sessions = {}
        
        for session in self.results['sessions']:
            pid = session['player_id']
            if pid not in player_sessions:
                player_sessions[pid] = []
            player_sessions[pid].append(session)
        
        stats = {}
        players_to_process = [player_id] if player_id else player_sessions.keys()
        
        for pid in players_to_process:
            if pid not in player_sessions:
                continue
                
            sessions = player_sessions[pid]
            all_scores = [score for session in sessions for score in session['scores']]
            all_steps = [step for session in sessions for step in session['steps']]
            
            stats[pid] = {
                'name': self.results['players'][pid]['name'],
                'experience_level': self.results['players'][pid]['experience_level'],
                'total_games': len(all_scores),
                'avg_score': float(np.mean(all_scores)),
                'max_score': int(np.max(all_scores)),
                'avg_steps': float(np.mean(all_steps))
            }
            
        return stats if player_id is None else stats.get(player_id)


def main():
    """Main function to run the human performance tracker"""
    tracker = HumanPerformanceTracker()
    
    while True:
        print("\nHuman Performance Tracker")
        print("------------------------")
        print("1. Add new player")
        print("2. Record play session")
        print("3. View player statistics")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == '1':
            name = input("Enter player name: ")
            player_id = name.lower().replace(' ', '_')
            try:
                exp_level = int(input("Enter experience level (1-10): "))
                if not 1 <= exp_level <= 10:
                    raise ValueError
            except ValueError:
                print("Invalid input. Using default experience level 5.")
                exp_level = 5
            
            tracker.add_player(player_id, name, exp_level)
            
        elif choice == '2':
            players = list(tracker.results['players'].keys())
            
            if not players:
                print("No players found. Please add a player first.")
                continue
                
            print("\nAvailable players:")
            for i, player_id in enumerate(players):
                print(f"{i+1}. {tracker.results['players'][player_id]['name']}")
                
            try:
                idx = int(input("\nEnter player number: ")) - 1
                if not 0 <= idx < len(players):
                    raise ValueError
                    
                player_id = players[idx]
                try:
                    num_games = int(input("Enter number of games to play (default: 5): ") or "5")
                except ValueError:
                    num_games = 5
                    
                tracker.record_session(player_id, num_games=num_games)
                
            except ValueError:
                print("Invalid selection.")
                
        elif choice == '3':
            stats = tracker.get_player_stats()
            
            if not stats:
                print("No player statistics available yet.")
                continue
                
            print("\nPlayer Statistics:")
            print("------------------")
            
            for pid, player_stats in stats.items():
                print(f"\nName: {player_stats['name']}")
                print(f"Experience: {player_stats['experience_level']}/10")
                print(f"Total Games: {player_stats['total_games']}")
                print(f"Average Score: {player_stats['avg_score']:.2f}")
                print(f"Maximum Score: {player_stats['max_score']}")
                print(f"Average Steps: {player_stats['avg_steps']:.2f}")
                
        elif choice == '4':
            print("Exiting Human Performance Tracker.")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
