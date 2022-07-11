import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# /*建行数据*/
df = pd.read_excel("datasets/short_sentences_220421分类.xlsx")

# 只保留负面情感的样本
raw = df.iloc[np.where(df['class1'].notnull())]
raw = raw.loc[30:200]
raw = raw[raw['class1'] != '不明']
raw = raw[['dataset', 'class1']]

# prepare train/valid/test
# todo add stratify
train, test = train_test_split(raw, test_size=0.2, random_state=2022)
train.to_csv('datasets/ccb/train.csv', index=False)
# 由于样本量太少，先用test来代替dev
test.to_csv('datasets/ccb/dev.csv', index=False)
test.to_csv('datasets/ccb/test.csv', index=False)

# /*线上购物数据*/
df = pd.read_csv('datasets/online_shopping_10_cats.csv')
df = df[['review', 'cat']]
train, dev_test = train_test_split(df, test_size=0.4, random_state=2022, stratify=df['cat'])
dev, test = train_test_split(dev_test, test_size=0.5, random_state=2022, stratify=dev_test['cat'])

train.to_csv('datasets/online_shopping/train.csv', index=False)
dev.to_csv('datasets/online_shopping/dev.csv', index=False)
test.to_csv('datasets/online_shopping/test.csv', index=False)
