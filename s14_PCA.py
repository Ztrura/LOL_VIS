import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
position_colors = {
    'TOP': 'red',
    'MID': 'blue',
    'ADC': 'green',
    'SUP': 'yellow',
    'JUG': 'gray'
}

data = pd.read_csv('data/s14/processed_s14_all.csv')
# data = pd.read_csv('data/s14/filtered_s14.csv')

# numeric_data = data.drop(columns=['Index', 'Player', 'Team', 'Position'])
numeric_data = data.drop(columns=['Index', 'Player', 'Team', 'Position', 'KDA', 'MVP', 'Appearances', 'Wins', 'Kills', 'Assists', 'Deaths', 'Damage Share'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

pca = PCA(n_components=2, whiten=False)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=pca_data[:, 0],
    y=pca_data[:, 1],
    hue=data['Position'],
    palette=position_colors,
    s=100
)

for i, player in enumerate(data['Player']):
    plt.text(pca_data[i, 0] + 0.02, pca_data[i, 1] + 0.02, player, fontsize=9)

plt.title('PCA Visualization of Player Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Position', loc='best')
# plt.savefig('results/s14/PCA_s14_filtered.png')
plt.savefig('results/s14/PCA_all.png')
plt.show()
