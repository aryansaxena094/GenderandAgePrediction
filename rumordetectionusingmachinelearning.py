# -*- coding: utf-8 -*-
"""rumordetectionusingmachinelearning

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dOubCjwftdm5Q-uZV-BHcKEIOvM1HuhF
"""

!pip install -q sklearn

import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv", header=None, dtype=str) #reading data
df['label'] = df[df.shape[1]-1] 
df.drop([df.shape[1]-2], axis=1, inplace=True) 
 
X = np.array(df.drop(['label'], axis = 1), dtype='<U13') #X contains all the values except the labels, converted to arrays
y = np.array(df['label'])  #Y Contains all the labels and not the body, and it has been converted to array
 
X = X[1:,2:] #X has been assigned columns, 1st columns contains the headlines, 2nd column contains the body
y = y[1:]   #Y has been assigned the only column which indicates the headlines
kf = KFold(n_splits=2) #Kfold Functions called
kf.get_n_splits(X)
y = list(map(int, y))
 
 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state= 100) #Data has been split into training and test size
 
        
X_train = list(X_train) #Conversion in the form of list
new_X_train = []
for i in X_train: #For loop has been used to append the values of the text into a list
    for j in i:
        new_X_train.append(j)     
        
X_test = list(X_test) #same for the testing data
new_X_test = []
for i in X_test:
    for j in i:
        new_X_test.append(j)
        
        
        
count_vect = CountVectorizer(lowercase=False)   #Count vectorizer has been called
X_train_counts = count_vect.fit_transform(new_X_train) #Training data list into count vector
X_test_counts = count_vect.transform(new_X_test) #Testing data list into count vector
 
 
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) #Tdidf Vectorizer has been called
tfidf_X_train = tfidf_vectorizer.fit_transform(new_X_train) #same steps
tfidf_X_test = tfidf_vectorizer.transform(new_X_test)
 
#funtion to plot confusion matrix 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.pink):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 
#function to find vectors that most affect labels
def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100): 
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]
 
    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)
 
    print()
 
    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)

most_informative_feature_for_binary_classification(count_vect, pac , n=30)

most_informative_feature_for_binary_classification(tfidf_vectorizer, pac , n=30)

#Random forest algorithm with count vectorizer
rf = RandomForestClassifier(n_estimators=400)
rf.fit(X_train_counts,y_train)
pred_y_rf = rf.predict(X_test_counts)
a= accuracy_score(y_test,pred_y_rf)
print ("Accuracy", float("{0:.2f}".format(a*100)))
cm_rf = confusion_matrix(y_test,pred_y_rf) 
plot_confusion_matrix(cm_rf, classes=['FAKE', 'REAL'])

#Random forest algorithm with tfidfVectorizer
rf.fit(tfidf_X_train,y_train)
pred_y_rf_tfidf = rf.predict(tfidf_X_test)
s= accuracy_score(y_test,pred_y_rf_tfidf)
print ("Accuracy", float("{0:.2f}".format(s*100)))
cm_rf_tfidf = confusion_matrix(y_test,pred_y_rf_tfidf) 
plot_confusion_matrix(cm_rf_tfidf, classes=['FAKE', 'REAL'])

#k nearest neighbour algorithm with count vectoriser
knn = KNeighborsClassifier()
knn.fit(X_train_counts,y_train)
pred_y_knn = knn.predict(X_test_counts)
b= accuracy_score(y_test,pred_y_knn)
print ("Accuracy", float("{0:.2f}".format(b*100)))
cm_knn = confusion_matrix(y_test,pred_y_knn) 
plot_confusion_matrix(cm_knn, classes=['FAKE', 'REAL'])

#k nearest neighbour algorithm with tfidfVectoriser
knn.fit(tfidf_X_train,y_train)
pred_y_knn_tfidf = knn.predict(tfidf_X_test)
r= accuracy_score(y_test,pred_y_knn_tfidf)
print ("Accuracy", float("{0:.2f}".format(r*100)))
cm_knn_tfidf = confusion_matrix(y_test,pred_y_knn_tfidf) 
plot_confusion_matrix(cm_knn_tfidf, classes=['FAKE', 'REAL'])

#SVM algorithm with tfidfVectorizer
svc = SVC(kernel='linear')
svc.fit(tfidf_X_train,y_train)
pred_y_svm_tfidf = svc.predict(tfidf_X_test)
p= accuracy_score(y_test,pred_y_svm_tfidf)
print ("Accuracy", float("{0:.2f}".format(p*100)))
cm_svm_tfidf = confusion_matrix(y_test,pred_y_svm_tfidf) 
plot_confusion_matrix(cm_svm_tfidf, classes=['FAKE', 'REAL'])

#SVM algorithm with count vectorizer
svc.fit(X_train_counts,y_train)
pred_y_svm = svc.predict(X_test_counts)
c= accuracy_score(y_test,pred_y_svm)
print ("Accuracy", float("{0:.2f}".format(c*100)))
cm_svm = confusion_matrix(y_test,pred_y_svm) 
plot_confusion_matrix(cm_svm, classes=['FAKE', 'REAL'])

#Passive Aggressive Classifier with count vectorizer
pac = PassiveAggressiveClassifier(max_iter=500)
pac.fit(X_train_counts,y_train)
pred_y_pac = pac.predict(X_test_counts)
d= accuracy_score(y_test,pred_y_pac)
print ("Accuracy", float("{0:.2f}".format(d*100)))
cm_pac = confusion_matrix(y_test,pred_y_pac)
plot_confusion_matrix(cm_pac, classes=['FAKE', 'REAL'])

#Passive Aggressive Classifier with tfidfVectorizer
pac.fit(tfidf_X_train,y_train)
pred_y_pac_tfidf = pac.predict(tfidf_X_test)
q= accuracy_score(y_test,pred_y_pac_tfidf)
print ("Accuracy", float("{0:.2f}".format(q*100)))
cm_pac_tfidf = confusion_matrix(y_test,pred_y_pac_tfidf)
plot_confusion_matrix(cm_pac_tfidf, classes=['FAKE', 'REAL'])