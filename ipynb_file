
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

##Loading Dataset

data = pd.read_csv("/content/diabetes.csv")



##Checking for missing values

sns.heatmap(data.isnull())

##Co- relation matrix

correlation = data.corr()
correlation

##As the correlation values are not clearly identified from here, plotting heat map can make it more clear

sns.heatmap(correlation)


##darker the colour lower the value of correlation

##Train_test_split

x = data.drop("Outcome",axis = 1)
#include all except outcome
y = data["Outcome"]
#include only outcome



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
x_train

x_test

y_train

y_test

##Training the model

model = LogisticRegression()
model.fit(x_train,y_train)



##Make Predictions

predictions = model.predict(x_test)
print(predictions)
[1 1 0 0 0 0 1 0 1 0 1 1 0 0 1 1 1 1 0 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 1 0
 1 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 0 1 1 0 0 0 0 1
 0 0 1 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1 1 0 0 1 0 1 0 1 0
 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 1
 0 0 1 1 0 0]

##Evaluation
##using logistic regression

accuracy = accuracy_score(predictions,y_test)
accuracy*100

# accuracy score using logistic regression
78.57142857142857


#3model training using svm

classifier = svm.SVC(kernel = "linear")
classifier.fit(x_train,y_train)

predictions_2 = classifier.predict(x_test)
predictions_2

array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,
       1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0,
       0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
       0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0])

accuracy_2 = accuracy_score(predictions_2,y_test)
accuracy_2*100

##accuracy score using support vector machine

79.87012987012987


##using random forest classifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(x_train,y_train)

predictions_3 = rf_model.predict(x_test)
predictions_3

array([0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0,
       1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
       0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0])

accuracy_3 = accuracy_score(predictions_3,y_test)

accuracy_3*100
##accuracy score using random forest classifier

78.57142857142857

##Using K_nearest_neighbours

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(x_train,y_train)

predictions_4 = knn_model.predict(x_test)
predictions_4

array([1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
       1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
       0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1])


accuracy_4 = accuracy_score(predictions_4,y_test)

accuracy_4

##accuracy score using K nearest neighbour
0.7272727272727273
