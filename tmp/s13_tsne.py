import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/s13/processed_s13.csv')
# data = pd.read_csv('data/filtered_s14.csv')

numeric_data = data.drop(columns=['Index', 'Player', 'Team', 'Position'])
# numeric_data = data.drop(columns=['Index', 'Player', 'Team', 'Position', 'MVP', 'Appearances', 'Wins', 'Kills', 'Assists', 'Deaths', 'Damage Share'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=20, n_iter=1000, init='pca', random_state=37)
tsne_data = tsne.fit_transform(scaled_data)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=tsne_data[:, 0],
    y=tsne_data[:, 1],
    hue=data['Position'],
    palette='Set2',
    s=100
)

for i, player in enumerate(data['Player']):
    plt.text(tsne_data[i, 0] + 0.02, tsne_data[i, 1] + 0.02, player, fontsize=9)

plt.title('t-SNE Visualization of Player Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Position', loc='best')
# plt.savefig('results/s14/tsne_s14_filtered.png')
plt.savefig('results/s13/tsne_s13.png')
plt.show()

