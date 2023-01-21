import os
from sklearn.metrics import *
import pandas as pd
import numpy as np

auc = []

for _ in range(5):
    os.system('./trainhmm -s 1.3.1 -m 1 -p 1 -e 0.000001,l hmm_scalable0.tsv model.txt predict.txt')
    os.system('./predicthmm -p 1 hmm_scalable1.tsv model.txt predict.txt')
    
    df = pd.read_csv('hmm_scalable1.tsv', sep = '\t', header = None)
    preds = pd.read_csv('predict.txt', sep = '\t', header = None)
    auc.append(roc_auc_score((2 - df[0]), preds[0]))

print(np.mean(auc), np.std(auc))
