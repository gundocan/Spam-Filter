from matplotlib import pyplot as plt
from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix, make_scorer, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
#Incase that NLTK dataset is not downloaded uncomment the following  two lines
#import nltk
#nltk.download('stopwords')

# Data Declaration
TR_DATA = 'spam-data-1'
TST_DATA = 'spam-data-2'

print("SPAM FILTER ACTIVATED")

#Accuracy Calculation using predefined modified accuracy
def modified_accuracy(y, y_pred):
    """Return a modified accuracy score with larger weight of false positives."""
    cm = confusion_matrix(y, y_pred)
    if cm.shape != (2, 2):
        raise ValueError('The ground truth values and the predictions may contain at most 2 values (classes).')
    return (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + 10 * cm[0, 1] + cm[1, 0])


our_scorer = make_scorer(modified_accuracy, greater_is_better=True)




# Helper function for CountVectorizer to analyze words and Stopwords are implemented to improve the accuracy
def my_analyzer(s):
    clean_mess = [word for word in s.split() if word.lower() not in stopwords.words('english')]
    tmp = []
    for word in clean_mess:
        if word.isalpha():
            if len(word)>2:
                tmp.append((word.strip()).lower())

    return tmp

#General function to train the models, pipeline is implemented for better results
def train_filter(X, y, clsf):
    """Return a trained spam filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert 'X' in locals().keys()
    assert 'y' in locals().keys()
    assert len(locals().keys()) == 3



    pipeline = ()
    parameters = {}
    if "MLPClassifier" in str(clsf):
        print("Multilayer Perteptron")
        parameters = {
            'vect__max_df': (0.75, 0.8, 0.85),
        }
        pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words = stopwords.words('english'))),
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('clf', clsf((100, 100, 80, 80, 40))),])
    elif "MultinomialNB" in str(clsf):
        print("Multinomial Naive Bayes")
        parameters = {
            'clf__alpha': (0.0001, 0.001, 0.01),
        }
        pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words = stopwords.words('english'))),
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('clf', clsf()),])
    else:
        print("AdaBoost")
        parameters = {
            'vect__max_df': (0.7, 0.8, 0.9),
            'clf__n_estimators': (100, 150, 200, 250),
        }
        pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words = stopwords.words('english'))),
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('clf', clsf(learning_rate=0.5)),])


    pipe = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    pipe.fit(X, y)
    #Find and Print  best parameters used for individual Classifiers
    best_param = pipe.best_estimator_.get_params()
    for name in sorted(parameters.keys()):
        print("%s: %r" % (name, best_param[name]))
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
    X_train = data_tr.data
    y_train = data_tr.target

    # Load testing data
    data_tst = load_files(TST_DATA, encoding='utf-8')
    X_test = data_tst.data
    y_test = data_tst.target

    #Data set is split into half
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.5)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_test, y_test, test_size=0.5)



    #Neural Network Classifier Test
    filter1 = train_filter(X_train2 + X_train1, np.concatenate((y_train2, y_train1)), MLPClassifier)
    y_tr_pred1 = predict(filter1, X_train)
    print('Modified accuracy on training data (1): ', modified_accuracy(y_train, y_tr_pred1))
    y_tst_pred1 = predict(filter1, X_test)
    print('Modified accuracy on testing data (1): ', modified_accuracy(y_test, y_tst_pred1))


    #Multinomial Naive Bayes  Classifier Test
    filter2 = train_filter(X_train1 + X_train2, np.concatenate((y_train1, y_train2)), MultinomialNB)
    y_tr_pred2 = predict(filter2, X_train)
    print('Modified accuracy on training data (2): ', modified_accuracy(y_train, y_tr_pred2))
    y_tst_pred2 = predict(filter2, X_test)
    print('Modified accuracy on testing data (2): ', modified_accuracy(y_test, y_tst_pred2))


    #AdoBoost Classifier test
    filter3 = train_filter(X_train1 + X_train2, np.concatenate((y_train1, y_train2)), AdaBoostClassifier)
    y_tr_pred3 = predict(filter3, X_train)
    print('Modified accuracy on training data (3): ', modified_accuracy(y_train, y_tr_pred3))
    y_tst_pred3 = predict(filter3, X_test)
    print('Modified accuracy on testing data (3): ', modified_accuracy(y_test, y_tst_pred3))

    #Plotting the ROC curve of the models and computing the AUC
    fpr1, tpr1, threshold1 = roc_curve(y_train, y_tr_pred1)
    auc_filter1 = auc(fpr1, tpr1)
    fpr2, tpr2, threshold2 = roc_curve(y_train, y_tr_pred2)
    auc_filter2 = auc(fpr2, tpr2)
    fpr3, tpr3, threshold3 = roc_curve(y_train, y_tr_pred3)
    auc_filter3 = auc(fpr3, tpr3)


    plt.figure(figsize=(5,5), dpi=100)
    plt.plot(fpr1, tpr1, linestyle ="-",color ="b" ,label =" MLPC (auc = %0.3f) " % auc_filter1 )
    plt.plot(fpr2, tpr2, linestyle="-",color = "g", label=" Multinomial Naive bayes (auc = %0.3f) " % auc_filter2)
    plt.plot(fpr3, tpr3, linestyle="-",color = "r" , label=" Adaboost (auc = %0.3f) " % auc_filter3)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel("false positive rate ")
    plt.ylabel("true positive rate ")
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend()
    plt.show()
