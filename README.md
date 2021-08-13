# Australian Rain Predictor: Project Overview
*   Created a tool that predicts if there will be rainfall in Australia (Accuracy ~ .87; AUC ~ .87) to communicate weather patterns and tendencies
*   Cleaned over 150,000 weather documentations from various cities across Australian
*   Built Logistic Regression, Random Forest, and SVM Classifier models
*   Tuned models by testing the efficacity of Normalizing and Resampling variables; Normalizing improved model performance and Resampling was deleterious
*   Optimized Logistic Regression and Random Forest Classifiers using Pipeline and GridsearchCV to reach the best model

# Code and Resources Used
**Python Version:** 3.9\
**Packages:**   pandas, numpy, sklearn, seaborn, pickle\

# Cleaning
Began cleaning and feature selection through the following changes:
*   Calculated the percent of missing data values for each column; removed columns in which nearly 40% of the data was missing
*   Dropped rows in which data values for the "RainTomorrow" column were missing because they were essential for model testing
*   Filled missing values in categorical columns with column averages
*   Determined and removed outliers in Rainfall column




