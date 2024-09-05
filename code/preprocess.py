import pandas as pd
import numpy as np

train_split = 0.8

def preprocess(data, impute_template):
    def train_test_split(data, train_split = train_split, skill_list = None):
        data = data.set_index(['user_id'])
        idx = np.random.permutation(data.index.unique())
        train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
        data_train = data.loc[train_idx].reset_index()
        data_test = data.loc[test_idx].reset_index()
        return data_train, data_test

    if 'skill_name' not in data.columns:
        data.rename(columns={'skill_id': 'skill_name'}, inplace=True)
    if 'original' in data.columns:
        data = data[data['original'] == 1]

    data['correct'] = data['correct'].round()

    data = data.groupby('user_id').filter(lambda q: len(q) > 1).copy()
    data['skill'], _ = pd.factorize(data['skill_name'], sort=True)
    data['skill_with_answer'] = data['skill'] * 2 + data['correct']

    data = data[~data['skill_name'].isna() & ~data['user_id'].isna() & ~data['correct'].isna() & (data['skill_name'] != 'Special Null Skill')]
    multi_col = 'template_id' if 'template_id' in data.columns else 'Problem Name'

    data_train, data_test = train_test_split(data)
    print("Train-test split finished...")

    train_skills = data_train['skill_name'].unique()
    skill_dict = {sn: i for i, sn in enumerate(train_skills)}
    print("Imputing skills...")
    repl = skill_dict[data_train['skill_name'].value_counts().index[0]]
    for skill_name in set(data_test['skill_name'].unique()) - set(skill_dict):
        skill_dict[skill_name] = repl

    print("Replacing skills...")
    data_train['skill_id'] = data_train['skill_name'].apply(lambda s: skill_dict[s])
    data_test['skill_id'] = data_test['skill_name'].apply(lambda s: skill_dict[s])

    if impute_template:
        train_templates = data_train[multi_col].unique()
        template_dict = {tn: i for i, tn in enumerate(train_templates)}
        print("Imputing templates...")
        repl = template_dict[data_train[multi_col].value_counts().index[0]]
        for temp_id in set(data_test[multi_col].unique()) - set(template_dict):
            template_dict[temp_id] = repl

        print("Replacing templates...")
        data_train[multi_col] = data_train[multi_col].apply(lambda s: template_dict[s])
        data_test[multi_col] = data_test[multi_col].apply(lambda s: template_dict[s])

    print("Number of embeddings:", data_test[multi_col].min(), data_test[multi_col].max(),
            data_test['skill_id'].min(), data_test['skill_id'].max())

    return *train_test_split(data_train, 0.9), data_test
