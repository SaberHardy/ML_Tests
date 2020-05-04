import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# import some data to play with
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url,names=names)

# """Data Preprocessing"""
# #divides data into labels and feature set
# X = dataset.iloc[:,0:4].values
# y = dataset.iloc[:,4].values
# #divides data into training and test sets:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# """Feature Scaling"""
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# """Performing LDA"""
# lda = LDA(n_components=1)
# X_train = lda.fit_transform(X_train,y_train)
# X_test = lda.transform(X_test)
# """Training and Making Predictions"""
# classifier = RandomForestClassifier(max_depth=2, random_state=0)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
#
# """Evaluating the Performance"""
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# print('Accuracy= ' + str(accuracy_score(y_test, y_pred)))
"""end algorithm of LDA"""
#-----------------------------------------------------




"""Preprocessing"""
#divide dataset into a feature set and corresponding labels

X = dataset.drop('Class',1) #featureset
y = dataset["Class"] #labels
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""Applying PCA"""
"""Performing PCA using Scikit-Learn is a two-step process"""
#1-initialize the PCA class by passing the number of
# components to the constructor.
#-----
#2- Call the fit and then transform methods
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#explained_variance_ratio_ which returns the
# variance caused by each of the principal components
# explained_variance = pca.explained_variance_ratio_
#
# print(explained_variance)
classifier = RandomForestClassifier(max_depth=2,random_state=0)
classifier.fit(X_train,y_train)

#predecting the test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + accuracy_score(y_test, y_pred))
"""end of PCA"""

