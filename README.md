# Australian Rain Predictor: Project Overview
*   Created a tool that predicts if there will be rainfall in Australia (Accuracy ~ .87; AUC ~ .87) to communicate weather patterns and tendencies
*   Cleaned over 150,000 weather documentations from various cities across Australian
*   Built Logistic Regression, Random Forest, and SVM Classifier models
*   Tuned models by testing the efficacity of Normalizing and Resampling variables; Normalizing improved model performance and Resampling was deleterious
*   Optimized Logistic Regression and Random Forest Classifiers using Pipeline and GridsearchCV to reach the best model

# Code and Resources Used
**Python Version:** 3.9\
**Packages:**   pandas, numpy, sklearn, seaborn, pickle
**Data Source:** https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

# Cleaning
Began cleaning and feature selection through the following changes:
*   Calculated the percent of missing data values for each column; removed columns where nearly 40% of the data was missing
*   Dropped rows in which data values for the RainTomorrow column were missing because they were essential for model testing
*   Filled missing values in categorical columns with column averages; filled missing values in categorical columns with "Unknown"
*   Determined and removed outliers in Rainfall column
*   Parsed out month from Date column
*   Label Encoded Month, Location, WindGustDir, WindDir9am, WindDir3pm, RainToday, and RainTomorrow columns

# EDA
I built histograms and bar charts to help understand our data and note any immediate trends. I also made a heatmap to make correlations limpid.

<img width="250" alt="location" src="https://user-images.githubusercontent.com/72672768/129390570-45730e4c-e8fa-4179-a359-e0dc8807d662.png">
<img width="250" alt="WindGustSpeed" src="https://user-images.githubusercontent.com/72672768/129390430-8cb6f788-e586-496e-a0f1-42fe3d811bb5.png">
<img width="250" alt="Temp3pm" src="https://user-images.githubusercontent.com/72672768/129390433-a9c59e75-31c2-44c9-9b73-616d25c84dce.png">
<img width="325" alt="heatmap" src="https://user-images.githubusercontent.com/72672768/129390117-5399c19d-f886-4ff6-b2ce-5e89b5a22098.png">

# Model Building
The first thing I did was split the data into train and test sets with a test size of 20%.

I used three different models and evaluated them with Accuracy and Area Under Curve. I trained the models on Normalized variables.

The models I tried:
  *   **Logistic Regression:** Benchmark for the model
  *   **Random Forest Classifier:** I thought this would be a good model to try because of the many categorical variables 
  *   **Support Vector Machine:** I wanted to try an SVM because this was a binary classification problem and frankly I was just curious because SVMs are such a classic classification algorithm

# Model Performance
The Random Forest model performed the best on the test and validation sets.
  *   **Random Forest Classifier:** Accuracy = .8667; AUC = .8670
  *   **Logistic Regression:** Accuracy = .8545; AUC = .8344
  *   **Support Vector Machine:** Accuracy = .8548; AUC = .8205
