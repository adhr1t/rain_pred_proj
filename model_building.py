import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE 


df = pd.read_csv('aus_rain_Fin.csv')

# train test split
X = df.drop('RainTomorrow', axis=1) 
Y = df.RainTomorrow.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Base Logistic Regression
reg = LogisticRegression(max_iter = 300)
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)


## Model Evaluation
# Confusion Matrix: heatmap of all 4 possible scenarios (false pos, true pos, etc.)
confMatrix = metrics.plot_confusion_matrix(reg, X_test, Y_test, cmap = 'coolwarm')  # true negative rate good, need to improve true pos

# Accuracy: the ratio of all correct predictions (true pos + true neg)/(all values)
metrics.accuracy_score(Y_test, Y_pred)  # Accuracy score is .8512

# ROC and AUC
Y_pred_prob = reg.predict_log_proba(X_test)[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_prob)
#plt.plot(fpr, tpr)
metrics.roc_auc_score(Y_test, Y_pred_prob)  # AUC score is .8221


## Model Tuning
# Variable Normalization; performed slightly better than base values
scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression with Normalized variables
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)

# Accuracy with Normalized variables
metrics.accuracy_score(Y_test, Y_pred)  # Accuracy score is .8545

# ROC and AUC
Y_pred_prob = reg.predict_log_proba(X_test)[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_prob)
#plt.plot(fpr, tpr)
metrics.roc_auc_score(Y_test, Y_pred_prob)  # AUC score is .8344


# Variable Resampling; performed significanlty worse than base and normalized values
# reset train and test
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#sm = SMOTE(random_state = 42)
#X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train.ravel())

# Logistic Regression with Resampled variables
#clf = LogisticRegression()
#model_res = clf.fit(X_train_res, Y_train_res)
#Y_pred_res = clf.predict(X_test)

# Accuracy with Resampled variables
#metrics.accuracy_score(Y_test, Y_pred_res)  # Accuracy score is .7476

# ROC and AUC
#Y_pred_prob = clf.predict_log_proba(X_test)[:,1]
#fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_prob)
#plt.plot(fpr, tpr)
#metrics.roc_auc_score(Y_test, Y_pred_prob)  # AUC score is .8180


## Random Forest Regression
