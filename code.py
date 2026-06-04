import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- 1. Load the dataset ---
df = pd.read_csv('diabetes.csv')

# --- 2. Handle invalid zero values ---
# Biologically, these features cannot be zero. Leaving them as 0 drastically skews the tree splits.
invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0 with NaN to accurately compute the median, then impute
df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)
df[invalid_zero_cols] = df[invalid_zero_cols].fillna(df[invalid_zero_cols].median())

# Prepare features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Train a Decision Tree classifier ---
# Limiting depth to 3 prevents overfitting and keeps the plotted tree readable
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# --- 4. Print accuracy and classification report ---
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- 5. Plot the decision tree ---
plt.figure(figsize=(16, 8))
plot_tree(clf, feature_names=X.columns, class_names=['Negative', 'Positive'], filled=True, rounded=True)
plt.title("Decision Tree: Pima Indians Diabetes Prediction")
plt.show()