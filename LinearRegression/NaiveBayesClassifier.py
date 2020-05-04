from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB


# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

#Label
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
#first step is convert the features into numbers

#create Label Encoder
label_encoder = preprocessing.LabelEncoder()
# Converting string labels into numbers.


# Converting string labels into numbers
weather_encoder = label_encoder.fit_transform(weather)
temp_encoded = label_encoder.fit_transform(temp)
label = label_encoder.fit_transform(play)
print("weat:", weather_encoder)
print("Temp:", temp_encoded)
print("Play:", label)

#Combinig weather and temp into single listof tuples

features = zip(weather_encoder, temp_encoded)
# features = features.reshape(-1,1)
model = GaussianNB()
features = list(features)
model.fit(features, label)
#prediction

predicted = model.predict([[1,2]])
print("tomorrow you will: ", predicted)

# -----------------------------------------------------------------
"""Naive Bayes with Multiple Labels"""
# #
# #Import scikit-learn dataset library
# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split#Load dataset
# from sklearn.naive_bayes import GaussianNB
# from sklearn import metrics
# wine = load_wine()
# print(f"Features: {wine.feature_names}")
# print(f"Labels: {wine.target_names}")
# print(wine.data.shape)
# #(178L, 13L)
# # print the wine data features (top 5 records)
# print(wine.data[0:5])
# # print the wine labels (0:Class_0, 1:class_2, 2:class_2)
# print(wine.target)
#
# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(
#     wine.data,
#     wine.target,
#     test_size=0.3,
#     random_state=109)
# #70% training and 30% test
# gnb = GaussianNB()
# #Train the model using the training sets
# gnb.fit(X_train,y_train)
# #Predict the response for test dataset
# y_pred = gnb.predict(X_test)
# print(f"the y_prediction= {y_pred}")
#
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))