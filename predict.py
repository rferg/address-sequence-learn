import helpers as h
import os
import pickle
import pandas as pd


def main():
    clf_name = input("Classifier filename:")
    clf_path = os.path.join('classifiers', clf_name)
    if(not(os.path.isfile(clf_path))):
        print("{} is not a file".format(clf_name))
        return
    else:
        with open(clf_path, 'rb') as f:
            clf = pickle.load(f)
        print('Classifier loaded')
        data = input('Data filename:')
        if(not(os.path.isfile(data))):
            print("{} is not a file".format(data))
        else:
            df = pd.read_csv(data, header=0, encoding='utf-8', dtype=str)
            data_name = os.path.splitext(os.path.basename(data))[0]
            txt_file_path = os.path.join('text_data_files', '{}.txt'.format(data_name))
            out_path = os.path.join('output', '{}_OUT.csv'.format(data_name))
            h.df_to_txt(df, txt_file_path)
            h.predict_and_save_output(txt_file_path, clf, df, out_path)
            print('Complete. Opening...')
            os.startfile(out_path)

if __name__ == '__main__':
     main()

