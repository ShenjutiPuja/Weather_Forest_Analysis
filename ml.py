import pandas as pd
data= pd.read_csv('C:/Users/User/OneDrive/Desktop/Weather_forest_analysis/Weather_Forest_Dataset.csv')

# Display basic information and the first few rows of the dataset
data_info = data.info()
data_head = data.head()

data_info, data_head
data.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Summary statistics
summary_stats = data.describe()

# Visualizations
plt.figure(figsize=(14, 8))

# Temperature distribution
plt.subplot(2, 2, 1)
sns.histplot(data['tem'], kde=True, bins=30, color='blue')
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Frequency')

# Rainfall distribution
plt.subplot(2, 2, 2)
sns.histplot(data['rain'], kde=True, bins=30, color='green')
plt.title('Rainfall Distribution')
plt.xlabel('Rainfall')
plt.ylabel('Frequency')

# Forest area over time
plt.subplot(2, 2, 3)
sns.lineplot(x='Year', y='forest_area', data=data, color='brown')
plt.title('Forest Area Over Time')
plt.xlabel('Year')
plt.ylabel('Forest Area')

# Pairplot for relationships
plt.subplot(2, 2, 4)
sns.scatterplot(x='tem', y='rain', data=data, hue='forest_area', palette='viridis')
plt.title('Temperature vs Rainfall')
plt.xlabel('Temperature')
plt.ylabel('Rainfall')

plt.tight_layout()
plt.show()

summary_stats


# Summary statistics
print(data.describe())

# Histograms
plt.figure(figsize=(14, 7))
plt.subplot(2, 2, 1)
sns.histplot(data['tem'], kde=True)
plt.title('Temperature Distribution')

plt.subplot(2, 2, 2)
sns.histplot(data['rain'], kde=True)
plt.title('Rainfall Distribution')

plt.subplot(2, 2, 3)
sns.histplot(data['forest_area'], kde=True)
plt.title('Forest Area Distribution')

plt.tight_layout()
plt.show()

# Line plots over time
plt.figure(figsize=(14, 7))
sns.lineplot(x='Year', y='forest_area', data=data)
plt.title('Forest Area Over Time')
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(x='Year', y='tem', data=data, label='Temperature')
sns.lineplot(x='Year', y='rain', data=data, label='Rainfall')
plt.title('Temperature and Rainfall Over Time')
plt.legend()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Box plots by month
plt.figure(figsize=(14, 7))
sns.boxplot(x='Month', y='tem', data=data)
plt.title('Temperature by Month')
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(x='Month', y='rain', data=data)
plt.title('Rainfall by Month')
plt.show()

# Correlation matrix
correlation_matrix = data.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

correlation_matrix


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare the data for modeling
X = data[['forest_area', 'Month', 'Year', 'rain']]
y = data['tem']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

mse, rmse, r2

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')

#Plot the actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Tempareture')
plt.ylabel('Predicted Tempareture')
plt.title('Actual vs. Predicted Tempareture')
plt.show()

# Get the coefficients
coefficients = model.coef_
intercept = model.intercept_

# Print the coefficients
print(f'Intercept: {intercept}')
print('Coefficients:')
for col, coef in zip(X.columns, coefficients):
    print(f'{col}: {coef}')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Split the data for polynomial features
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)

# Train the models on the polynomial features
rf_model.fit(X_train_poly, y_train_poly)
gb_model.fit(X_train_poly, y_train_poly)
ridge_model.fit(X_train_poly, y_train_poly)
lasso_model.fit(X_train_poly, y_train_poly)

# Make predictions on the test set
rf_pred = rf_model.predict(X_test_poly)
gb_pred = gb_model.predict(X_test_poly)
ridge_pred = ridge_model.predict(X_test_poly)
lasso_pred = lasso_model.predict(X_test_poly)

# Evaluate models using R-squared
rf_r2 = r2_score(y_test_poly, rf_pred)
gb_r2 = r2_score(y_test_poly, gb_pred)
ridge_r2 = r2_score(y_test_poly, ridge_pred)
lasso_r2 = r2_score(y_test_poly, lasso_pred)


print(f'Random Forest: {rf_r2}')
print(f'Gradient Boosting: {gb_r2}')
print(f'Ridge Regression: {ridge_r2}')
print(f'Lasso Regression: {lasso_r2}')


import matplotlib.pyplot as plt

# Model accuracies
model_accuracies = {
    "Random Forest": 95.34,
    "Gradient Boosting": 95.85,
    "Ridge Regression": 94.07,
    "Lasso Regression": 93.89
}

# Convert to DataFrame for plotting
accuracy_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=accuracy_df, palette='viridis')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracies')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features and target variable
X = data[['forest_area', 'Month', 'Year', 'rain']]
y = data['tem']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize models
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# Dictionary to store evaluation results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}

results_df = pd.DataFrame(results).T
print(results_df)


import matplotlib.pyplot as plt

# Plot RMSE for each model
plt.figure(figsize=(10, 6))
plt.bar(results_df.index, results_df['RMSE'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('RMSE of Different Models')
plt.xticks(rotation=45)
plt.show()

# Plot R² for each model
plt.figure(figsize=(10, 6))
plt.bar(results_df.index, results_df['R2'], color='lightgreen')
plt.xlabel('Model')
plt.ylabel('R²')
plt.title('R² of Different Models')
plt.xticks(rotation=45)
plt.show()

import pickle
from sklearn.ensemble import GradientBoostingRegressor

# Assuming you have already trained your model

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save the model to disk
with open('temperature_model.pkl', 'wb') as f:
    pickle.dump(model, f)
