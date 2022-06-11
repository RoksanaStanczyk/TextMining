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
    x = df['text_original']
    y = df['category']
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
    #   output: SVM Accuracy Score:  97.90419161676647
    """Opis:
    Klasyfikator SVM to jeden z najpopularniejszych algorytmów uczenia z nadzorem, który jest wykorzystywany do 
    problemów klasyfikacji. Celem opisywanego algorytmu jest stworzenie takiej lini (granicy decyzyjnej), która podzieli 
    próbki na odpowiednie klasy, tak by następnie z łatwością umieszczać kolejne próbki do odpowiedniej klasy. 
    Do zwizualizowania poprawności predykcji modelu, wygenerowano Confusion Matrix, który wskazuje, czy model SVM 
    poprawnie klasyfikuje kategorie do poszczególnych klas.
    Największą trudność dla klasyfikatora była klasą "politics", gdyż 2 przypadki sklasyfikował do klasy business, 
    dwa do klasy sport i po jednym przypadku do pozosyałych klas: entertainmet i tech. 
    Aby poprawnie zinterpretować wyniki klasyfkacji wygenerowano również Classification report, który wskazuje precyzje 
    poszczególnych klas, czyli zdolność klasyfikatora do nieoznaczenia próbki negatywnej jako pozytywnej. Najwyższą 
    precyzją okazała się klasa sport, która wynosi 0.99, gdzie 1 jest wynikiem najwyższym. Zgadza się ten wynik 
    z wynikiem przedstawionym w Confusion Matrix, ponieważ w klasie sport występił tylko 1 przypadek przyporządkowania 
    do innej klasy - business.
    Następnym parametrem generowanym przez Classification report jest recall, czyli zdolność klasyfikatora 
    do znalezienia wszystkich próbek pozytywnych. Najwyższą wartością charakteryzuje się klasa business oraz 
    sport - 0.99. 
    Kolejnym parametrem jest f1-score, czyli ważona śrdnia harmoniczna precision i recall, która jest najwyższa dla 
    klasy sport i osiaga wartość 0.99. 
    """


def training_RandomForestClassifier(df: pd.DataFrame):
    x = df['text_original']
    y = df['category']
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(x, y, test_size=0.3, random_state=1)
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    Train_X_transform = vectorizer.fit_transform(Train_X)
    Test_X_transform = vectorizer.transform(Test_X)

    # creating a RF classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(Train_X_transform, Train_Y)
    y_pred = clf.predict(Test_X_transform)
    print("RandomForestClassifier Accuracy Score: ", metrics.accuracy_score(y_pred, Test_Y) * 100)
    print(classification_report(Test_Y, y_pred, labels=clf.classes_))
    cm = confusion_matrix(Test_Y, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()


#   output RandomForestClassifier Accuracy Score:  98.05389221556887

def training_LogisticRegression(df: pd.DataFrame):
    x = df['text_original']
    y = df['category']
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
    # output LogisticRegression Accuracy Score: 97.75449101796407
