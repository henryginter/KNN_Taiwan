#KNN

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('default of credit card clients.xls')

# Data preprocessing
new_header = dataset.iloc[0]
dataset = dataset[1:] #take the data less the header row
dataset.columns = new_header #set the header row as the df header
dataset.SEX = dataset.SEX.replace(2, 0) #made the column binary
dataset.MARRIAGE = dataset.MARRIAGE.replace(0, 3) #removing abnomalies
y = dataset.iloc[:, 23].values.astype('int64') #extracting the dependant outcome

#making separate binary variable columns from 3 MARRIAGE categories (onehotencoder)
onehot = pd.get_dummies(dataset['MARRIAGE']) 
dataset = dataset.drop(['MARRIAGE'], axis =1)
dataset = dataset.drop(['default payment next month'], axis =1)
dataset = dataset.join(onehot)

# Feature Scaling non-binary variables
from sklearn.preprocessing import StandardScaler
scaled_features = dataset.copy()
col_names = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features
X = scaled_features.values.astype('float') #extracting independent variables

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

#Trying out different odd K values
from sklearn.neighbors import KNeighborsClassifier
neighbors = list(range(1,50,2)) 
cv_scores = []
for k in neighbors: 
    classifier = KNeighborsClassifier(n_neighbors = k,
                                  metric = 'minkowski',
                                  p = 2)
    classifier.fit(X_train, y_train) # Fitting classifier to the Training set
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    error_rate = (FP+FN)/(TP+TN+FP+FN)
    cv_scores.append(error_rate)

#Printing out the optimal K value and plot the results
optimal_k = neighbors[cv_scores.index(min(cv_scores))]
print ("The optimal number of neighbors is %d" % optimal_k + " with an error rate of %.3f" % min(cv_scores))
plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Error Rate')
plt.show()

