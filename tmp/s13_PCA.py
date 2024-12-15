import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/s13/processed_s13.csv')
# data = pd.read_csv('data/filtered_s13.csv')

numeric_data = data.drop(columns=['Index', 'Player', 'Team', 'Position'])
# numeric_data = data.drop(columns=['Index', 'Player', 'Team', 'Position', 'MVP', 'Appearances', 'Wins', 'Kills', 'Assists', 'Deaths', 'Damage Share'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

pca = PCA(n_components=2, whiten=False)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=pca_data[:, 0],
    y=pca_data[:, 1],
    hue=data['Position'],
    palette='Set2',
    s=100
)

for i, player in enumerate(data['Player']):
    plt.text(pca_data[i, 0] + 0.02, pca_data[i, 1] + 0.02, player, fontsize=9)

plt.title('PCA Visualization of Player Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Position', loc='best')
# plt.savefig('results/s14/PCA_s14_filtered.png')
plt.savefig('results/s13/PCA_s13.png')
plt.show()
