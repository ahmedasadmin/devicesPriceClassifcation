import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import SVC classifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score   # import metrics to compute accuracy

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print(train_data)
print(train_data.columns)
print(train_data['price_range'].value_counts())          # check if dataset balanced or not 
print(train_data.info())
print(train_data.isnull().sum())                         # check null values
train_data.dropna(inplace=True)
print(train_data.isnull().sum())                         # check null values


print(test_data)
print(test_data.columns)
print(test_data.info())
print(test_data.isnull().sum())                         # check null values
print(round(train_data.describe(),2) )                  # summary statistics

X = train_data.drop(['price_range'], axis=1)
y = train_data['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


scaler = StandardScaler()                              # scalling features 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


                                                        # instantiate classifier with default hyperparameters
svc=SVC() 
                                                        # fit classifier to training set
svc.fit(X_train,y_train)
                                                        # make predictions on test set
y_pred=svc.predict(X_test)

                                                        # compute and print accuracy score
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))