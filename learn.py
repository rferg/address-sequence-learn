from seqlearn.evaluation import whole_sequence_accuracy
from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import helpers as h
import pickle



def main():
    df = pd.read_csv('full_data.csv', header=0, encoding='utf-8', dtype=str)
    df_train, df_test = train_test_split(df, test_size=0.2)
    h.df_to_txt(df_train, 'full_training_data.txt')
    h.df_to_txt(df_test, 'full_testing_data.txt')
    X_train, y_train, lengths_train = h.load_conll('full_training_data.txt', 
                                                   h.features)
    X_test, y_test, lengths_test = h.load_conll('full_testing_data.txt', h.features)
    clf = StructuredPerceptron(decode='viterbi', lr_exponent=0.1, verbose=True, 
                               max_iter=25)
    clf.fit(X_train, y_train, lengths_train)
    y_pred = clf.predict(X_test, lengths_test)
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("Whole sequence accuracy: {}".format(whole_sequence_accuracy(y_test, 
          y_pred, lengths_test)))
    response = input('OK to pickle? [y/n]')
    if response == 'y':
        name = input('Name:')
        with open('classifiers/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(clf, f)
        print('Complete.')
    else:
        print('Classifier discarded. Complete.')
    return

if __name__ == '__main__':
    main()
    

    
    
        
    
    
