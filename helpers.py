import csv
import re
from seqlearn.datasets import load_conll
import pandas as pd


def conll_label(row):
    row_labels = []
    for col in row.index:
        if not(pd.isnull(row.loc[col])):
            for word in str(row.loc[col]).split(' '):
                row_labels.append([word, col])
    return row_labels

def select_columns(df):
    df_new = df[['PREFIX', 'FIRSTNAME', 'MIDDLENAME', 'LASTNAME', 'SUFFIX', 
        'STREET', 'ADDR1', 'CITY', 'STATE', 'ZIP']]
    return df_new

def df_to_txt(df, file_name):
    df = select_columns(df)
    list_ = df.apply(conll_label, axis=1).tolist()
    with open(file_name, 'w') as f:
        for row in list_:
            for tup in row:
                f.write("{} POS {}\n".format(tup[0], tup[1]))
            f.write("\n")
    f.close()

def get_address_reference():
    with open('name_address_reference.csv', 'r') as f:
        reader = csv.reader(f)
        ref_list = list(reader)
    prefixes = ref_list[0]
    suffixes = ref_list[1]
    streets = ref_list[2]
    units = ref_list[3]
    states = ref_list[4]
    pobox = ref_list[5]
    directions = ref_list[6]
    return prefixes, suffixes, streets, units, states, pobox, directions

def tag(word, prefixes, suffixes, streets, units, states, pobox, directions):
    #read in address reference lists
    #remove punctuation and trim whitespaces
    word.replace('.', '').replace(',', '').replace('-', '').strip()
    if bool(re.search(r'\d', word)):
        #if word contains number
        t = 'N'
    elif word in prefixes:
        t = 'F'
    elif word in suffixes:
        t = 'X'
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
    prefixes, suffixes, streets, units, states, pobox, directions = get_address_reference()
    word = row[i]
    yield "word:{}".format(word.lower())
    
    yield "tag:{}".format(tag(word.lower(), prefixes, suffixes, streets, units, 
               states, pobox, directions))
    yield "length:{}".format(len(word))
    
    if i > 0:
        yield "tag-1:{}".format(tag(row[i-1], prefixes, suffixes, streets, units,
                     states, pobox, directions))
        yield "word-1:{}".format(row[i-1])
        if i > 1:
            yield "tag-2:{}".format(tag(row[i-2], prefixes, suffixes, streets,
                         units, states, pobox, directions))
            yield "word-2:{}".format(row[i-2])

            
    if i + 1 < len(row):
        yield "tag+1:{}".format(tag(row[i+1], prefixes, suffixes, streets, 
                     units, states, pobox, directions))
        yield "word+1:{}".format(row[i+1])
        if i + 2 < len(row):
            yield "tag+2:{}".format(tag(row[i+2], prefixes, suffixes, streets, 
                         units, states, pobox, directions))
            yield "word+2:{}".format(row[i+2])
            
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
        new_row = {'PREFIX':'', 'FIRSTNAME':'', 'MIDDLENAME':'', 'LASTNAME':'', 'SUFFIX':'',
                   'STREET':'', 'ADDR1':'', 'CITY':'', 'STATE':'', 'ZIP':''}
        for col in ['PREFIX', 'FIRSTNAME', 'MIDDLENAME', 'LASTNAME', 'SUFFIX', 
        'STREET', 'ADDR1', 'CITY', 'STATE', 'ZIP']:
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
            
def predict_and_save_output(file_name, clf, orig_df, out_path,
                            columns = ['PREFIX', 'FIRSTNAME', 'MIDDLENAME', 'LASTNAME', 'SUFFIX', 
        'STREET', 'ADDR1', 'CITY', 'STATE', 'ZIP']):
    X, _, lengths = load_conll(file_name, features)
    y_pred = clf.predict(X, lengths)
    new_df = get_new_df(orig_df, y_pred, lengths)
    new_df = new_df[columns]
    new_df.to_csv(out_path, index=False)
