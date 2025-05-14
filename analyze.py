from scipy.stats import f_oneway
import pandas as pd

# Load the dataset
df = pd.read_csv('results.csv')

# Group the data by 'Pop Size' and 'Mut Rate' and extract the last 10 'Avg Score' values for each group
score_groups = [group['Avg Score'].values[-10:] for _, group in df.groupby(['Pop Size', 'Mut Rate'])]

# Check if there are at least two groups
if len(score_groups) < 2:
    print("Error: At least two groups are required for ANOVA. Check your dataset.")
else:
    # Perform ANOVA
    f_stat, p_value = f_oneway(*score_groups)
    print(f"ANOVA: F={f_stat:.2f}, p={p_value:.4f}")