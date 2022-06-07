import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from data_cleaning import text_tokenizer
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression


def training_svm(df: pd.DataFrame):
    x = df['text']
    y = df['airline_sentiment']
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(x, y, test_size=0.3, random_state=1)
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    Train_X_transform = vectorizer.fit_transform(Train_X)
    Test_X_transform = vectorizer.transform(Test_X)

    sklearn.ensemble.GradientBoostingClassifier(verbose=1)
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_transform, Train_Y)
    predictions_SVM = SVM.predict(Test_X_transform)
    print("SVM Accuracy Score: ", accuracy_score(predictions_SVM, Test_Y) * 100)
    # target_names = ['sport', '', 'class 2']
    print(classification_report(Test_Y, predictions_SVM, labels=SVM.classes_))
    cm = confusion_matrix(Test_Y, predictions_SVM, labels=SVM.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SVM.classes_)
    disp.plot()
    plt.show()
    # SVM Accuracy Score:  76.57103825136612


def training_RandomForestClassifier(df: pd.DataFrame):
    x = df['text']
    y = df['airline_sentiment']
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(x, y, test_size=0.3, random_state=1)
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    Train_X_transform = vectorizer.fit_transform(Train_X)
    Test_X_transform = vectorizer.transform(Test_X)

    # creating a RF classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(Train_X_transform, Train_Y)
    # performing predictions on the test dataset
    y_pred = clf.predict(Test_X_transform)
    # metrics are used to find accuracy or error
    # using metrics module for accuracy calculation
    print("RandomForestClassifier Accuracy Score: ", metrics.accuracy_score(y_pred, Test_Y) * 100)
    print(classification_report(Test_Y, y_pred, labels=clf.classes_))
    cm = confusion_matrix(Test_Y, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()


#    RandomForestClassifier Accuracy Score:  75.3415300546448

def training_LogisticRegression(df: pd.DataFrame):
    x = df['text']
    y = df['airline_sentiment']
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(x, y, test_size=0.3, random_state=1)
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    Train_X_transform = vectorizer.fit_transform(Train_X)
    Test_X_transform = vectorizer.transform(Test_X)

    logreg = LogisticRegression(max_iter=1000, random_state=0).fit(Train_X_transform, Train_Y)
    y_pred = logreg.predict(Test_X_transform)
    print("LogisticRegression Accuracy Score: ", metrics.accuracy_score(y_pred, Test_Y) * 100)
    print(classification_report(Test_Y, y_pred, labels=logreg.classes_))
    cm = confusion_matrix(Test_Y, y_pred, labels=logreg.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
    disp.plot()
    plt.show()
    # LogisticRegression Accuracy Score:  77.41347905282332

# opis klasyfikatorów:
# Najlepszym klasyfikatorem okazał się LogisticRegression (accuracy: 77.41347905282332)
# ConfusionMatrix modelu LogisticRegression wskazał, że wszystkie klasyfikacje wykonuje prawidłowo,
# natomiast najgorzej radzi sobie w przypadku klasyfikacji tweetów do kategorii neutral, gdyż te często przypisuje
# do klasy negatywnej(na 936 neutralnych tweetów, aż 321 przypisał do negatywnego sentymentu.)
# Wygenerowany raport klasyfikacji wskazał, że najwyższą precyzję klasyfikacji ma klasa negative (0.84), następnie
# positive (0.73), a neutral (0.60)
# Wartośc recall w raporcie klasyfikacji wskazuje stosunek liczby prawdziwych trafień do liczby wyników fałszywie dodatnich,
# klasa negative posiada najwyższą jego wartość (0.88)
# Wynik F1 to średnia precyzji klasa negative posiada najwyższą jego wartość (0.86)
