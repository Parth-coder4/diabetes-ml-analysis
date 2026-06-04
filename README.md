# Pima Indians Diabetes Predictor

This repository contains a machine learning classification project that predicts the onset of diabetes using the Pima Indians Diabetes Dataset. 

##  Project Overview

* **Dataset:** Pima Indians Diabetes Dataset (`diabetes.csv`)
* **Model:** Decision Tree Classifier
* **Objective:** Predict whether a patient has diabetes based on diagnostic measures such as Glucose, BMI, Insulin, and Blood Pressure.

##  Key Features & Implementation

* **Data Preprocessing:** Handled invalid biological zero values (e.g., a Blood Pressure of 0) by replacing them with `NaN` and imputing the median values to prevent skewed model training.
* **Model Training:** Trained a Decision Tree classifier, explicitly limiting `max_depth=3` to prevent overfitting and maintain interpretability.
* **Evaluation:** Outputs model accuracy and a detailed classification report (Precision, Recall, F1-score).
* **Visualization:** Generates a complete, visual map of the tree's decision-making logic using `matplotlib` and `sklearn.tree.plot_tree`.

## Tech Stack

* **Language:** Python 
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Data Visualization:** `matplotlib`
* <img width="1536" height="850" alt="Figure_1" src="https://github.com/user-attachments/assets/ecf0c9c0-2561-479c-9b9f-c5051588a9aa" />
