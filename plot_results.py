import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('results.csv')
for (pop, mut), group in df.groupby(['Pop Size', 'Mut Rate']):
    plt.plot(group['Generation'], group['Avg Score'], label=f"Pop={pop}, Mut={mut}")
plt.xlabel('Generation')
plt.ylabel('Average Score')
plt.legend()
plt.savefig('score_curves.png')