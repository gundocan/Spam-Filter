"""
B(E)3M33UI - Support script for the first semestral task
"""
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix, make_scorer, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from collections import Counter
#Incase that NLTK dataset is not downloaded uncomment the following  two lines
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB




# Data Declaration
TR_DATA = 'spam-data-1'
TST_DATA = 'spam-data-2'

#Accuracy Calculation using predefined modified accuracy
def modified_accuracy(y, y_pred):
    """Return a modified accuracy score with larger weight of false positives."""
    cm = confusion_matrix(y, y_pred)
    if cm.shape != (2, 2):
        raise ValueError('The ground truth values and the predictions may contain at most 2 values (classes).')
    return (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + 10 * cm[0, 1] + cm[1, 0])


our_scorer = make_scorer(modified_accuracy, greater_is_better=True)

#General function to train the models, pipeline is implemented for better results
def train_filter(X, y):
    """Return a trained spam filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert 'X' in locals().keys()
    assert 'y' in locals().keys()
    assert len(locals().keys()) == 2


    parameters = {
        'vect__max_df': (0.70, 0.75, 0.8, 0.85),
    }
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words = stopwords.words('english'))),
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('clf', MLPClassifier((100,100,80,80,40))), ])

    pipe = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
    pipe.fit(X, y)
    return pipe

#Predicting Model results
def predict(filter, X):
    """Produce predictions for X using given filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert len(locals().keys()) == 2

    return filter.predict(X)





if __name__ == '__main__':

    # Demonstration how the filter will be used but you can do whatever you want with the these data
    # Load training data
    data_tr = load_files(TR_DATA, encoding='utf-8')
    #print(str(data_tr))
    X_train = data_tr.data
    #print("X_train : ",str( X_train))
    y_train = data_tr.target
    #print("y_train : ", y_train)

    # Load testing data
    data_tst = load_files(TST_DATA, encoding='utf-8')
    #Inıtialize input and output
    X_test = data_tst.data
    y_test = data_tst.target

    #Taking %50 of training dataset and concatanated with test dataset
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.5)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_test, y_test, test_size=0.5)
    filter1 = train_filter(X_train2 + X_train1, np.concatenate((y_train2, y_train1)))
    y_tr_pred = predict(filter1, X_train)
    print('Modified accuracy on training data: ', modified_accuracy(y_train, y_tr_pred))
    y_tst_pred = predict(filter1, X_test)
    print('Modified accuracy on testing data: ', modified_accuracy(y_test, y_tst_pred))


    # Plotting the ROC curve
    fpr  , tpr , threshold =roc_curve(y_train, y_tr_pred)
    auc_filter = auc(fpr , tpr)
    #plt.figure(figsize=(8,8), dpi=100)
    plt.plot(fpr, tpr, marker ="." ,lw= 2,label =" MLPC (auc = %0.3f) " % auc_filter )
    plt.xlabel("false positive rate ")
    plt.ylabel("true positive rate ")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic (ROC)')
    plt.show()


""" Taking %100 of training dataset (not ideal)

    filter2 = train_filter(X, y)
    y_tr_pred = predict(filter2, X_train)
    print('Modified accuracy on training data: ', modified_accuracy(y_train, y_tr_pred))
    y_tst_pred = predict(filter2, X_test)
    print('Modified accuracy on testing data: ', modified_accuracy(y_test, y_tst_pred))
"""




 







