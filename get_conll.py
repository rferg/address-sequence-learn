import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('addr_data.csv', header=0, encoding='utf-8', dtype=str)
df_train, df_test = train_test_split(df, test_size=0.2)

def conll_label(row):
    row_labels = []
    for col in row.index:
        if not(pd.isnull(row.loc[col])):
            for word in str(row.loc[col]).split(' '):
                row_labels.append((word, col))
    return row_labels

list_train = df_train.apply(conll_label, axis=1).tolist()
list_test = df_test.apply(conll_label, axis=1).tolist()
with open('training_data.txt', 'w') as f:
    for row in list_train:
        for tup in row:
            f.write("{} POS {}\n".format(tup[0], tup[1]))
        f.write("\n")

with open('testing_data.txt', 'w') as f:
    for row in list_test:
        for tup in row:
            f.write("{} POS {}\n".format(tup[0], tup[1]))
        f.write("\n")
f.close()
