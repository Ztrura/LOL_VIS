import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/s13/processed_s13.csv')
# data = pd.read_csv('data/filtered_s14.csv')

# numeric_data = data.drop(columns=['Index', 'Player', 'Team', 'Position'])
numeric_data = data.drop(columns=['Index', 'Player', 'Team', 'Position', 'MVP', 'Appearances', 'Wins', 'Kills', 'Assists', 'Deaths', 'Damage Share'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

import umap

# umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=37)
# umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=1.0, random_state=37)
# umap_reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=1.0, random_state=37)
# umap_reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=37)

# umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=37)
umap_reducer = umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.1, random_state=37)
umap_data = umap_reducer.fit_transform(scaled_data)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=umap_data[:, 0],
    y=umap_data[:, 1],
    hue=data['Position'],
    palette='Set2',
    s=100
)

for i, player in enumerate(data['Player']):
    plt.text(umap_data[i, 0] + 0.02, umap_data[i, 1] + 0.02, player, fontsize=9)

plt.title('UMAP Visualization of Player Data')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.legend(title='Position', loc='best')
# plt.savefig('results/s14/umap_s14_filtered.png')
plt.savefig('results/s13/umap_s13.png')
plt.show()
