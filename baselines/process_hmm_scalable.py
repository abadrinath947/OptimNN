from preprocess import *
import pandas as pd
from tqdm import tqdm
import re

data = pd.read_csv('data/processed_algebra.csv', encoding = 'latin')

for i, df in enumerate(preprocess(data, False)):
    df['correct'] = 2 - df['correct']
    if 'skill_name' not in df.columns:
        df['skill_name'] = df['skill_id']
    df['dummy'] = 0
    df = df[['correct', 'user_id', 'dummy', 'skill_name']]

    df.to_csv(f'../hmm-scalable/hmm_scalable{i}.tsv', index = False, header = False, sep = '\t')
