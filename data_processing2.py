import pandas as pd
import re

data = pd.read_csv('data/s14/processed_s14.csv')

filtered_data = data[data['Position'].isin(['TOP', 'MID', 'ADC'])]

output_file = 'data/s14/filtered_s14.csv'
filtered_data.to_csv(output_file, index=False)