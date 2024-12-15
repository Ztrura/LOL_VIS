import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors
position_colors = {
    'TOP': 'red',
    'MID': 'blue',
    'ADC': 'green',
    'SUP': 'yellow',
    'JUG': 'gray'
}

data = pd.read_csv('data/s14/processed_s14_all.csv')
# data = pd.read_csv('data/filtered_s14.csv')

# numeric_data = data.drop(columns=['Index', 'Player', 'Team', 'Position'])
numeric_data = data.drop(columns=['Index', 'Player', 'Team', 'Position', 'KDA', 'MVP', 'Appearances', 'Wins', 'Kills', 'Assists', 'Deaths', 'Damage Share'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=20, n_iter=1000, init='pca', random_state=37)
tsne_data = tsne.fit_transform(scaled_data)

# import umap
#
# umap_reducer = umap.UMAP(n_components=2, n_neighbors=77, min_dist=0.5, random_state=37)
# umap_data = umap_reducer.fit_transform(scaled_data)

plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x=tsne_data[:, 0],
    y=tsne_data[:, 1],
    hue=data['Position'],
    palette=position_colors,
    s=100
)

cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect("add")
def on_add(sel):
    index = sel.index
    sel.annotation.set_text(data['Player'].iloc[index])
    sel.annotation.set_fontsize(10)
    sel.annotation.set_backgroundcolor("#FFEECC")
    sel.annotation.set_bbox(dict(
        boxstyle="round,pad=0.3",
        edgecolor="gray",
        facecolor="#FFEECC",
        alpha=0.8
    ))

plt.title('tSNE Visualization of Player Data')
plt.xlabel('tSNE Component 1')
plt.ylabel('tSNE Component 2')
plt.legend(title='Position', loc='best')
plt.savefig('results/s14/tsne_new.png')
plt.show()
