import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


df = pd.read_csv('aus_rain_Fin.csv')

# train test split
X = df.drop('RainTomorrow', axis=1) 
Y = df.RainTomorrow.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# Logistic Regression
reg = LogisticRegression(max_iter = 300)
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)


## Model Evaluation
# Confusion Matrix: heatmap of all 4 possible scenarios (false pos, true pos, etc.)
confMatrix = metrics.plot_confusion_matrix(reg, X_test, Y_test, cmap = 'coolwarm')  # true negative rate good, need to improve true pos

# Accuracy: the ratio of all correct predictions (true pos + true neg)/(all values)
metrics.accuracy_score(Y_test, Y_pred)

# ROC and AUC
Y_pred_prob = reg.predict_log_proba(X_test)[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_prob)
plt.plot(fpr, tpr)
metrics.roc_auc_score(Y_test, Y_pred_prob)  # AUC score is .8237
