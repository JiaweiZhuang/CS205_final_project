import numpy as np
import pandas as pd
import math
import random

df= pd.read_csv("data/covtype_data.csv", header=None)

#normalize
# X = df.iloc[:, :-1]
# X_normed = X / X.max(axis=0)
# Y = df.iloc[: ,-1]
# X_normed = X_normed.fillna(0)

# X_normed.to_csv("data/observation_full.csv", index=False)
# Y.to_csv("data/label_full.csv", index=False, header=True)




sample = pd.DataFrame(columns = df.columns)
#choose 10% for each group 
for i in range(1,8):
    subset = df[df[54]==i]
    rows = random.sample(subset.index, int(subset.shape[0]*0.05))
    sample = sample.append(df.ix[rows],ignore_index=True)


#normalize
X = sample.iloc[:, :-1]
X_normed = X / X.max(axis=0)
Y = sample.iloc[: ,-1]
X_normed = X_normed.fillna(0)

X_normed.to_csv("data/observation_sample.csv", index=False)

Y.to_csv("data/label_sample.csv", index=False, header=True)