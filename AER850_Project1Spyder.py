''' Step #1: Data Processing'''

import pandas as pd

# Reading dataset
df = pd.read_csv("data/Project_1_Data.csv")

# Looking at first few rows of dataset
print(df.head())
print(df.info())
print(df.describe())

# Splitting data into features and target variables (X and Y)
X = df[['X', 'Y', 'Z']] 
y = df['Step']           

'''Step #2: Data Visualization'''

import matplotlib.pyplot as plt

#Calculating stats for X, Y, and Z
step_stats_X = df.groupby('Step')['X'].agg(['mean', 'std', 'min', 'max']).rename(columns=lambda col: 'X_' + col)
step_stats_Y = df.groupby('Step')['Y'].agg(['mean', 'std', 'min', 'max']).rename(columns=lambda col: 'Y_' + col)
step_stats_Z = df.groupby('Step')['Z'].agg(['mean', 'std', 'min', 'max']).rename(columns=lambda col: 'Z_' + col)
step_stats_XYZ = pd.concat([step_stats_X, step_stats_Y, step_stats_Z], axis=1)
pd.set_option('display.max_columns', None)
print(step_stats_XYZ)
print(df.groupby('Step')['Y'].unique())

# Creating figure with a specified size 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.title("3D Scatter Plot")

# Creating 3D scatter plot using the coordinate columns from dataframe 
scatter = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis', alpha=0.7)

# Setting the labels for the X, Y and Z axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adding colourbar to plot
fig.colorbar(scatter, ax=ax, label='Step',shrink=0.5, pad=0.15)
plt.show()

'''Step #3: Correlation Analysis'''

import seaborn as sns

# Computting correlation matrix
data_for_matrix = df[['X', 'Y', 'Z', 'Step']]

#Calculating and display correlation maxtrix using "Pearson" correlation
correlation_matrix = data_for_matrix.corr(method='pearson')
print("Correlation Matrix:")
print(correlation_matrix)

# Displaying the correlation matrix using heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title("Correlation Heatmap of X, Y, Z relating to Step")
plt.show()

''' Step #4: Classification Model Development/Engineering'''

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Splitting data into training and testing sets
sets = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sets.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Combing through data to ensure no missing values
print(df.isnull().sum()) 

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL TRAINING  

# Model 1: Logistic Regression + Grid Search CV
log_reg = LogisticRegression(max_iter=1000)
param_grid_log_reg = {'C': [0.01, 0.1, 1, 10, 100]}
grid_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5)  
grid_log_reg.fit(X_train_scaled, y_train)
print(f"Logistic Regression Parameters: {grid_log_reg.best_params_}\n")

# Model 2: Random Forest Classifier + Grid Search CV
rf = RandomForestClassifier()
param_dist_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf':[1, 2, 4]}
grid_rf = GridSearchCV(rf, param_dist_rf, cv=5)
grid_rf.fit(X_train_scaled, y_train)
print(f"Random Forest Parameters: {grid_rf.best_params_}\n")

# Model 3: Support Vector Classifier + Grid Search CV
svc = SVC()
param_grid_svc = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_svc = GridSearchCV(svc, param_grid_svc, cv=5)  
grid_svc.fit(X_train_scaled, y_train)
print(f"SVC Parameters: {grid_svc.best_params_}\n")

# Model 4: Decision Tree Classifier + Randomized Search CV
dt = DecisionTreeClassifier()
param_dist_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
random_search_dt = RandomizedSearchCV(dt, param_distributions=param_dist_dt, n_iter=10, cv=5)
random_search_dt.fit(X_train_scaled, y_train)
print(f"Decision Tree Parameters: {random_search_dt.best_params_}\n")


''' Step #5: Model Performance Analysis'''

from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Comparing performance of models based on accuracy, precision, and f1 score
models = {
    'Logistic Regression': grid_log_reg,
    'Random Forest': grid_rf,
    'SVC': grid_svc,
    'Decision Tree': random_search_dt
}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"Model: {name}\n")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}\n")
    print(f"F1 score: {f1_score(y_test, y_pred, average='weighted'):.4f}\n")
 
 # Plotting confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (6, 4))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
       
''' Step #6: Stacked Model Performance Analysis'''

import numpy as np
from sklearn.ensemble import StackingClassifier

# Defining base estimators using the best estimators from previous models
base_estimators = [
    ('random_forest', grid_rf.best_estimator_),   
    ('svc', grid_svc.best_estimator_)
]

final_estimator = LogisticRegression(max_iter=1000)

# Creating StackingClassifier
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_estimator,
    cv=5
)

# Fitting stacking classifier on scaled training data
stacking_clf.fit(X_train_scaled, y_train)

# Evaluating stacked model
y_pred_stacked = stacking_clf.predict(X_test_scaled)

# Modelling performance metrics
print(f"Stacked Classifier accuracy: {accuracy_score(y_test, y_pred_stacked):.4f}\n")
print(f"Stacked Classifier precision: {precision_score(y_test, y_pred_stacked, average='weighted'):.4f}\n")
print(f"Stacked Classifier f1 Score: {f1_score(y_test, y_pred_stacked, average='weighted'):.4f}\n")

# Plotting confusion matrix for Stacked Classifier
conf_matrix_stacked = confusion_matrix(y_test, y_pred_stacked)
plt.figure(figsize=(6, 4))  
sns.heatmap(conf_matrix_stacked, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Stacked Classifier")
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


''' Step #7: Model Evaluation'''

import joblib

# Saving stacked classifier model
model_filename = 'stacked_classifier_model.joblib'
joblib.dump(stacking_clf, model_filename)
print(f"Model saved as {model_filename}\n")

# Loading the saved model
loaded_model = joblib.load(model_filename)

# Defining new coordinates to predict maintenance steps
random_coordinates = np.array([[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]])

# Scaling the new data 
random_coordinates_scaled = scaler.transform(random_coordinates)

# Predicting maintenance steps 
predicted_steps = loaded_model.predict(random_coordinates_scaled)

# Displaying predictions for each set of coordinates
for i, coords in enumerate(random_coordinates):
    print(f"Coordinates: {coords} ===> Predicted maintenance step: {predicted_steps[i]}\n")

