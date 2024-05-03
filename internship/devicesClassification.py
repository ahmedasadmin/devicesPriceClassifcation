#########################################################################
# Prices Classification of Devices Based on its SPecs 
# author: Ahmed Muhammed Abdelgaber
##########################################################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC                   # import SVC classifier
from sklearn.metrics import accuracy_score   # import metrics to compute accuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
#############################################################################
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

target_labels = ['low cost', 'medium cost', 'high cost', 'very high cost']
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
svc=SVC(kernel='linear', degree=3, random_state=0) 
                                                        # fit classifier to training set
svc.fit(X_train,y_train)
test_data_values = test_data.values                                                        # make predictions on test set
reshaped_test_data = test_data_values.reshape((-1,20))
y_pred=svc.predict(X_test)  
print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3]))
# Prediction for first 10 devices 
y_pred10 = svc.predict(reshaped_test_data[:10])
y_pred10 = map(lambda x: target_labels[x], y_pred10)
# confussion matrix 
cm = confusion_matrix(y_test, y_pred, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
print('Model accuracy score with rbf kernel: {0:0.4f}'. format(accuracy_score(y_test, y_pred))) # compute and print accuracy score
print("first 10 devices test :", list(y_pred10))

disp.plot()
plt.show()
