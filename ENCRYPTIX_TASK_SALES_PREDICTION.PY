import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "advertising.csv"
df = pd.read_csv(file_path)

# Display the first few rows and basic information about the dataset
print("First few rows of the dataset:")
print(df.head().to_string())
print("\nDataset information:")
print(df.info())
print("\n")

# Separate features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train_poly, y_train)

# Make predictions using the test set
y_pred = model.predict(X_test_poly)

# Print model coefficients
coefficients = pd.DataFrame(model.coef_, index=poly.get_feature_names_out(X.columns), columns=['Coefficient'])
print("Model Coefficients:")
print(coefficients.to_string())
print("\nIntercept:", model.intercept_)
print()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evaluation_metrics = pd.DataFrame({'Metric': ['Mean Squared Error', 'R-squared'], 'Value': [mse, r2]})
print("Evaluation Metrics:")
print(evaluation_metrics.to_string())
print()

# Plotting scatter plots for each advertising medium vs. Sales
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(df['TV'], df['Sales'], color='blue')
plt.title('TV Advertising vs. Sales')
plt.xlabel('TV Advertising ($)')
plt.ylabel('Sales')

plt.subplot(1, 3, 2)
plt.scatter(df['Radio'], df['Sales'], color='green')
plt.title('Radio Advertising vs. Sales')
plt.xlabel('Radio Advertising ($)')
plt.ylabel('Sales')

plt.subplot(1, 3, 3)
plt.scatter(df['Newspaper'], df['Sales'], color='orange')
plt.title('Newspaper Advertising vs. Sales')
plt.xlabel('Newspaper Advertising ($)')
plt.ylabel('Sales')

plt.tight_layout()
plt.show()

# Pair plot for overall visualization
sns.pairplot(df, height=3)
plt.suptitle('Pair Plot of Advertising vs. Sales', y=1.02)
plt.show()

# Plotting actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.grid(True)
plt.show()
