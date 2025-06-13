import pickle

# Path to the .pkl file
file_path = "experiment_data/results/enhanced_model_final_eval_34games.pkl"

# Open and load the pickled file
try:
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
except FileNotFoundError:
    print(f"File {file_path} not found. Please check the path.")
    exit()

# Print the contents to inspect the structure
print("Contents of enhanced_model_final_eval_34games.pkl:")
print(results)

# If it's a dictionary, print its keys to understand the structure
if isinstance(results, dict):
    print("\nDictionary Keys:")
    print(list(results.keys()))