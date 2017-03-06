import csv
import re
from seqlearn.datasets import load_conll
from seqlearn.evaluation import whole_sequence_accuracy
from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('addr_data.csv', header=0, encoding='utf-8', dtype=str)
df_train, df_test = train_test_split(df, test_size=0.2)

def conll_label(row):
    row_labels = []
    for col in row.index:
        if not(pd.isnull(row.loc[col])):
            for word in str(row.loc[col]).split(' '):
                row_labels.append([word, col])
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

def get_address_reference():
    with open('address_reference.csv', 'r') as f:
        reader = csv.reader(f)
        ref_list = list(reader)
    streets = ref_list[0]
    units = ref_list[1]
    states = ref_list[2]
    pobox = ref_list[3]
    directions = ref_list[4]
    return streets, units, states, pobox, directions

def tag(word, streets, units, states, pobox, directions):
    #read in address reference lists
    #remove punctuation and trim whitespaces
    word.replace('.', '').replace(',', '').replace('-', '').strip()
    if bool(re.search(r'\d', word)):
        #if word contains number
        t = 'N'
    elif word in streets:
        t = 'R'
    elif word in units:
        t = 'U'
    elif word in states:
        t = 'S'
    elif word in pobox:
        t = 'P'
    elif word in directions:
        t = 'D'
    else:
        t = 'L'
    return t

def features(row, i):
    streets, units, states, pobox, directions = get_address_reference()
    word = row[i]
    yield "word:{}".format(word.lower())
    
    yield "tag:{}".format(tag(word.lower(), streets, units, states, pobox, directions))
    yield "length:{}".format(len(word))
    
    if i > 0:
        yield "tag-1:{}".format(tag(row[i-1], streets, units, states, pobox, directions))
        yield "word-1:{}".format(row[i-1])
        if i > 1:
            yield "tag-2:{}".format(tag(row[i-2], streets, units, states, pobox, directions))
            yield "word-2:{}".format(row[i-2])

            
    if i + 1 < len(row):
        yield "tag+1:{}".format(tag(row[i+1], streets, units, states, pobox, directions))
        yield "word+1:{}".format(row[i+1])
        if i + 2 < len(row):
            yield "tag+2:{}".format(tag(row[i+2], streets, units, states, pobox, directions))
            yield "word+2:{}".format(row[i+2])

    


X_train, y_train, lengths_train = load_conll('training_data.txt', features)
X_test, y_test, lengths_test = load_conll('testing_data.txt', features)
clf = StructuredPerceptron(decode='viterbi', lr_exponent=0.1, verbose=True, max_iter=15)
clf.fit(X_train, y_train, lengths_train)
y_pred = clf.predict(X_test, lengths_test)
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("Whole sequence accuracy: {}".format(whole_sequence_accuracy(y_test, y_pred, lengths_test)))





def get_new_df(df, y_pred, lengths):
    count = 0
    preds = []
    for i in lengths:
        preds.append(y_pred[count:count+i])
        count += i
    df.reset_index(drop=True, inplace=True)
    dfjoin = pd.concat([df, pd.DataFrame(preds)], axis=1)
    def move_by_pred(row):
        split = []
        new_row = {'STREET':'', 'ADDR1':'', 'CITY':'', 'STATE':'', 'ZIP':''}
        for col in ['STREET', 'ADDR1', 'CITY', 'STATE', 'ZIP']:
            if not(pd.isnull(row.loc[col])):
                for word in str(row.loc[col]).split(' '):
                    split.append(word)
        for i, word in enumerate(split):
            current = new_row[row.iloc[i + 5]]
            if current == '':
                new_row[row.iloc[i + 5]] = current + word
            else:
                new_row[row.iloc[i + 5]] = current + ' ' + word
        new_row = pd.Series(new_row, dtype=str)
        return new_row
    new_df = dfjoin.apply(move_by_pred, axis=1)
    return new_df
    
new_df = get_new_df(df_test, y_pred, lengths_test)
new_def = new_df[['STREET', 'ADDR1', 'CITY', 'STATE', 'ZIP']]
new_def.to_csv('output.csv', index=False)            
df_test.to_csv('test.csv', index=False)    

    
    
        
    
    
