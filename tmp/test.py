import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/s14/s14.csv')
data['Player'] = data['Player'].str.split(' ').str[0]

def extract_kda(kda_str):
    match = re.search(r'(\d+\.\d+) KDA: (\d+\.\d+) / (\d+\.\d+) / (\d+\.\d+)', kda_str)
    if match:
        kda = float(match.group(1))
        kills = float(match.group(2))
        deaths = float(match.group(3))
        assists = float(match.group(4))
        return kda, kills, deaths, assists
    else:
        return None, None, None, None

data[['KDA', 'Kill', 'Death', 'Assist']] = data['KDA'].apply(lambda x: pd.Series(extract_kda(x)))

percent_columns = ['Teamfight Participation Rate', 'Damage Share', 'Damage Taken Share']
for col in percent_columns:
    data[col] = data[col].str.rstrip('%').astype(float) / 100.0

numeric_data = data.drop(columns=['Player', 'Position'])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# 降维
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# 聚类
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(pca_data)
data['Cluster'] = clusters

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=pca_data[:, 0],
    y=pca_data[:, 1],
    hue=data['Position'],
    style=data['Cluster'],
    palette='Set2',
    s=100
)

for i, player in enumerate(data['Player']):
    plt.text(pca_data[i, 0] + 0.02, pca_data[i, 1] + 0.02, player, fontsize=9)

plt.title('Player Clustering with Position Highlight')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Position', loc='best')
plt.savefig('results/result_initial.png')
plt.show()
