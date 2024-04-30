import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("csv.csv")


plt.figure(figsize=(10, 6))
plt.plot(df['Step'], df['Value'], linestyle='-')

plt.xlabel('Iterations')
plt.ylabel('Rewards')

plt.grid(True)
plt.tight_layout()
plt.savefig("Rewards.png")