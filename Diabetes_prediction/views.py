from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    data = pd.read_csv("/Users/amitrawat/AmitRawat/Diabetic management System/diabetes.csv")
    x = data.drop("Outcome", axis=1)
    # include all except outcome
    y = data["Outcome"]
    # include only outcome
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""

    if pred == [1]:
        result1 = "positive"
    else:
        result1 = "negative"
    return render(request, 'predict.html', {"result2": result1})
