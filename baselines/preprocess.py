import pandas as pd
import numpy as np

train_split = 0.8

def preprocess(data, impute_template):
    def train_test_split(data, skill_list = None):
        np.random.seed(42)
        data = data.set_index(['user_id', 'skill_name'])
        idx = np.random.permutation(data.index.unique())
        train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
        data_train = data.loc[train_idx].reset_index()
        data_val = data.loc[test_idx].reset_index()
        return data_train, data_val

    if 'skill_name' not in data.columns:
        data.rename(columns={'skill_id': 'skill_name'}, inplace=True)
    if 'original' in data.columns:
        data = data[data['original'] == 1]

    data = data[~data['skill_name'].isna() & (data['skill_name'] != 'Special Null Skill')]
    multi_col = 'template_id' if 'template_id' in data.columns else 'Problem Name'

    data_train, data_val = train_test_split(data)
    print("Train-test split finished...")

    train_skills = data_train['skill_name'].unique()
    skill_dict = {sn: i for i, sn in enumerate(train_skills)}
    print("Imputing skills...")
    repl = skill_dict[data_train['skill_name'].value_counts().index[0]]
    for skill_name in set(data_val['skill_name'].unique()) - set(skill_dict):
        skill_dict[skill_name] = repl

    print("Replacing skills...")
    data_train['skill_id'] = data_train['skill_name'].apply(lambda s: skill_dict[s])
    data_val['skill_id'] = data_val['skill_name'].apply(lambda s: skill_dict[s])

    if impute_template:
        train_templates = data_train[multi_col].unique()
        template_dict = {tn: i for i, tn in enumerate(train_templates)}
        print("Imputing templates...")
        repl = template_dict[data_train[multi_col].value_counts().index[0]]
        for temp_id in set(data_val[multi_col].unique()) - set(template_dict):
            template_dict[temp_id] = repl

        print("Replacing templates...")
        data_train[multi_col] = data_train[multi_col].apply(lambda s: template_dict[s])
        data_val[multi_col] = data_val[multi_col].apply(lambda s: template_dict[s])

    print("Number of embeddings:", data_val[multi_col].min(), data_val[multi_col].max(),
            data_val['skill_id'].min(), data_val['skill_id'].max())

    return data_train, data_val
