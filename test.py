'''STEP 1 - DATA PROCESSING'''
import pandas as pd

# Reading the dataset
df = pd.read_csv("data/Project_1_Data.csv")

# Checking the first few rows of the dataset
print(df.head())

# Splitting data into features (X) and target variable (y)
X = df[['X', 'Y', 'Z']]  # Features: X, Y, Z coordinates
y = df['Step']           # Target: Step (maintenance step)

# Check for class imbalance
print(f"\nClass Distribution: \n{y.value_counts(normalize=True)}")


'''STEP 2 - SPLITTING DATA INTO TRAIN AND TEST SETS'''
from sklearn.model_selection import StratifiedShuffleSplit

# Splitting Data into Train and Test Sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Check the distribution of steps in train/test sets
print(f"\nTrain set step distribution: \n{y_train.value_counts(normalize=True)}")
print(f"Test set step distribution: \n{y_test.value_counts(normalize=True)}")


'''STEP 3 - DATA VISUALIZATION'''
import matplotlib.pyplot as plt

# Creating a new figure with a specified size, and adding a 3D subplot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(projection='3d')

# Creating a 3D scatter plot using the 'X', 'Y', and 'Z' columns from the DataFrame (df)
# The points are colored based on the values in the 'Step' column.
scatter = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='plasma', vmin=1, vmax=12)

# Setting the labels for the axis to X, Y and Z
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adding a colorbar to the plot
plt.colorbar(scatter, label='Step')
plt.title('3D Scatter plot of X, Y, Z colored by Step')
plt.show()


'''STEP 4 - CORRELATION ANALYSIS'''
import seaborn as sns
# Compute the correlation matrix
# The correlation matrix helps us understand how features like X, Y, Z are related to the target (Step)
correlation_matrix = df[['X', 'Y', 'Z', 'Step']].corr()

# Display the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, linewidths=0.5)
plt.title("Correlation Matrix of Features and Target")
plt.show()


'''STEP 5 - DATA CLEANING & PREPROCESSING'''
from sklearn.preprocessing import StandardScaler
# Checking for missing values
print(df.isnull().sum())  # Ensure no missing values

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


'''STEP 6 - MODEL TRAINING'''
# We will train 3 models: Logistic Regression, Support Vector Classifier (SVC), and Random Forest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

# Model 1: Logistic Regression with Grid Search CV
log_reg = LogisticRegression(max_iter=1000)
param_grid_log_reg = {'C': [0.01, 0.1, 1, 10, 100]}
grid_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5)  
grid_log_reg.fit(X_train_scaled, y_train)
print(f"Best Logistic Regression Parameters: {grid_log_reg.best_params_}")

# Model 2: Support Vector Classifier (SVC) with Grid Search CV
svc = SVC()
param_grid_svc = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_svc = GridSearchCV(svc, param_grid_svc, cv=5)  
grid_svc.fit(X_train_scaled, y_train)
print(f"Best SVC Parameters: {grid_svc.best_params_}")

# Model 3: Random Forest Classifier with Grid Search CV
rf = RandomForestClassifier()
param_dist_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf':[1, 2, 4]}
grid_rf = GridSearchCV(rf, param_dist_rf, cv=5)
grid_rf.fit(X_train_scaled, y_train)
print(f"Best Random Forest Parameters: {grid_rf.best_params_}")

# Model 4: Decision Tree Classifier with Randomized Search CV
dt = DecisionTreeClassifier()
param_dist_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
random_search_dt = RandomizedSearchCV(dt, param_distributions=param_dist_dt, n_iter=10, cv=5)
random_search_dt.fit(X_train_scaled, y_train)
print(f"Best Decision Tree Parameters: {random_search_dt.best_params_}")


'''STEP 7 - MODEL PERFORMANCE ANALYSIS'''
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
# Compare the performance of all models based on Accuracy, Precision, and F1 Score
models = {
    'Logistic Regression': grid_log_reg,
    'SVC': grid_svc,
    'Random Forest': grid_rf,
    'Decision Tree': random_search_dt
}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(conf_matrix).plot()
    plt.title(f"Confusion Matrix for {name}")
    plt.show()