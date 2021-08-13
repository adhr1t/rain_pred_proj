import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pickle


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
# Random Forest with Normalized variables; base variables are miniscually worse
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
Y_pred_rfc = rfc.predict(X_test)

# Confusion Matrix
confusion_matrix(Y_test, Y_pred_rfc)
confMatrix = metrics.plot_confusion_matrix(rfc, X_test, Y_test, cmap = 'coolwarm')  # true negative rate good, need to improve true pos

# Classification Report; Accuracy included in this and confirmed by cross validation
print(classification_report(Y_test, Y_pred_rfc))

# Accuracy 
rfc_cv_acc_score = np.mean(cross_val_score(rfc, X_train, Y_train, cv=3, scoring= 'accuracy'))    # Accuracy score is .8615

# ROC and AUC
rfc_cv_AUC_score = np.mean(cross_val_score(rfc, X_train, Y_train, cv=3, scoring= 'roc_auc'))    # AUC score is .8628



## Standard Vector Machine; far too inefficient to be used. Accuracy and AUC scores were both lower than random forest's
# SVM with Normalized variables
#svmc = SVC(kernel='rbf', random_state = 42)
#svmc.fit(X_train, Y_train)
#Y_pred_SVM = svmc.predict(X_test)

# Confusion Matrix
#confusion_matrix(Y_test, Y_pred_SVM)
#confMatrix = metrics.plot_confusion_matrix(svmc, X_test, Y_test, cmap = 'coolwarm')  # true negative rate good, need to improve true pos

# Classification Report; Accuracy included in this and confirmed by cross validation
#print(classification_report(Y_test, Y_pred_SVM))

# Accuracy 
#svmc_cv_acc_score = np.mean(cross_val_score(svmc, X_train, Y_train, cv=3, scoring= 'accuracy'))    # Accuracy score is .8548

# ROC and AUC
#svmc_cv_AUC_score = np.mean(cross_val_score(svmc, X_train, Y_train, cv=3, scoring= 'roc_auc'))      # AUC score is .8205



## Grid Search of best performing algorithms: Logistic Regression and Random Forest
# Create first pipeline for base without reducing features.

pipe = Pipeline([('classifier' , RandomForestClassifier())])
# pipe = Pipeline([('classifier', RandomForestClassifier())])

# Create param grid
param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : range(10,300,10),
    'classifier__max_features' : ('auto','sqrt','log2')}
]

# Create grid search object
gs = GridSearchCV(pipe, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_gs = gs.fit(X_train, Y_train)
gs.best_score_      # 
gs.best_estimator_  # n_estimators = 170


# Classification Report
classes = best_gs.predict(X_test)
print(metrics.classification_report(classes, Y_test))

# Accuracy
best_gs.score(X_test, Y_test)   # Accuracy score is .8667

# ROC and AUC
probs = best_gs.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)     # AUC score is .8670



## Pickle model 
# doing this incase we want to build an API later or just so we don't have to train and cross validate the models again
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open('model_file' + ".p", "wb" ))

# test if pickled model actually works
file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
    
for i in range(len(df)):
    print(model.predict(X_test[i].reshape(1,-1)))    # test the model to make sure it works