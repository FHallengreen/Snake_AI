import pickle

# Path to the .pkl file
file_path = "experiment_data/best_ai/best_model.pkl"

# Open and load the pickled file
try:
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
except FileNotFoundError:
    print(f"File {file_path} not found. Please check the path.")
    exit()

# Print the contents to inspect the structure
print("Contents of best_model.pkl:")
print(results)

# If it's a dictionary, print its keys to understand the structure
if isinstance(results, dict):
    print("\nDictionary Keys:")
    print(list(results.keys()))