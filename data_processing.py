import pandas as pd
import re

data = pd.read_csv('data/lgc/lgc.csv')

def extract_kda(kda_str):
    # match = re.search(r'(\d+\.\d+) KDA: (\d+\.\d+) / (\d+\.\d+) / (\d+\.\d+)', kda_str)
    match = re.search(r'(\d+\.\d+) KDA(\d+\.\d+) / (\d+\.\d+) / (\d+\.\d+)', kda_str)
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

data['Team'] = data['Player'].str.split(' ').str[1]
data['Player'] = data['Player'].str.split(' ').str[0]

output_file = 'data/lgc/processed_lgc.csv'
data.to_csv(output_file, index=False)