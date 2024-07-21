import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv(r'titanic\Titanic-Dataset.csv')

# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(data.head())

# Display basic statistics
print("\nBasic statistics of the dataset:")
print(data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Convert categorical variables to numerical ones
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'])

# Feature engineering: Create new features
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = 1  # Initialize to 1 (is alone)
data['IsAlone'].loc[data['FamilySize'] > 1] = 0  # If family size is greater than 1 then not alone

# Drop irrelevant columns
data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# Visualize the data

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Create survival rate by gender plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=data)
plt.title("Survival Rate by Gender")
plt.show()

# Create survival rate by class plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title("Survival Rate by Class")
plt.show()

# Create age distribution plot
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], kde=True, bins=30)
plt.title("Age Distribution")
plt.show()

# Create fare distribution plot
plt.figure(figsize=(8, 6))
sns.histplot(data['Fare'], kde=True, bins=30)
plt.title("Fare Distribution")
plt.show()

# Create family size distribution plot
plt.figure(figsize=(8, 6))
sns.histplot(data['FamilySize'], kde=True, bins=10)
plt.title("Family Size Distribution")
plt.show()

# Define features and target variable
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Try different models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'\n{name} Model Evaluation:')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))

# Hyperparameter tuning for the best model (Random Forest in this case)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("\nBest parameters found for Random Forest:")
print(grid_search.best_params_)

# Final model evaluation with the best parameters
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print('\nTuned Random Forest Model Evaluation:')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
