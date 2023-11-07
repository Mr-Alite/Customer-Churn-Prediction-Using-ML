import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
# Step 1: Load and preprocess the dataset
data = pd.read_csv("E:\Churndata\churn-data-1.csv")
X = data[['Total intl calls', 'Total intl charge', 'Customer service calls']]
y = data['Churn']
data
# Step 2: Divide the data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
# Step 3: Construct different regression algorithms
linear_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor()
random_forest_model = RandomForestRegressor()
lasso_model = Lasso()
bayesian_linear_model = BayesianRidge()
stepwise_model = LinearRegression()
rfe = RFE(stepwise_model, n_features_to_select=3)
rfe.fit(X_train, y_train)
# Model Training
X_train_stepwise = X_train[X_train.columns[rfe.support_]]
X_test_stepwise = X_test[X_test.columns[rfe.support_]]

# Fit the models on the training data
linear_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
bayesian_linear_model.fit(X_train, y_train)
stepwise_model.fit(X_train_stepwise, y_train)
#calculate matrices for random forest
random_forest_pred = random_forest_model.predict(X_test)
random_forest_mse = mean_squared_error(y_test, random_forest_pred)
random_forest_rmse = np.sqrt(random_forest_mse)
random_forest_mae = mean_absolute_error(y_test, random_forest_pred)
random_forest_r2 = r2_score(y_test, random_forest_pred)

# Step 4: Calculate different regression metrics on the test set
linear_pred = linear_model.predict(X_test)
print(linear_pred)
decision_tree_pred = decision_tree_model.predict(X_test)
print(decision_tree_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
print(linear_mse)
decision_tree_mse = mean_squared_error(y_test, decision_tree_pred)
print(decision_tree_mse)
linear_rmse = np.sqrt(linear_mse)
print(linear_rmse)
decision_tree_rmse = np.sqrt(decision_tree_mse)
print(decision_tree_rmse)
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mae  
decision_tree_mae = mean_absolute_error(y_test, decision_tree_pred)
decision_tree_mae
linear_r2 = r2_score(y_test, linear_pred)
linear_r2
decision_tree_r2 = r2_score(y_test, decision_tree_pred)
decision_tree_r2
random_forest_pred = random_forest_model.predict(X_test)
print(random_forest_pred)
lasso_pred = lasso_model.predict(X_test)
print(lasso_pred) 
random_forest_mse = mean_squared_error(y_test, random_forest_pred)
print(random_forest_mse)  
lasso_mse = mean_squared_error(y_test, lasso_pred)
print(lasso_mse)
random_forest_rmse = np.sqrt(random_forest_mse)
print(random_forest_rmse)
lasso_rmse = np.sqrt(lasso_mse)
print(lasso_rmse)
random_forest_mae = mean_absolute_error(y_test, random_forest_pred)
random_forest_mae
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_mae
random_forest_r2 = r2_score(y_test, random_forest_pred)
random_forest_r2
lasso_r2 = r2_score(y_test, lasso_pred)
lasso_r2
bayesian_linear_pred = bayesian_linear_model.predict(X_test)
bayesian_linear_mse = mean_squared_error(y_test, bayesian_linear_pred)
bayesian_linear_rmse = np.sqrt(bayesian_linear_mse)
bayesian_linear_mae = mean_absolute_error(y_test, bayesian_linear_pred)
bayesian_linear_r2 = r2_score(y_test, bayesian_linear_pred)
stepwise_pred = stepwise_model.predict(X_test_stepwise)
stepwise_mse = mean_squared_error(y_test, stepwise_pred)
stepwise_rmse = np.sqrt(stepwise_mse)
stepwise_mae = mean_absolute_error(y_test, stepwise_pred)
stepwise_r2 = r2_score(y_test, stepwise_pred)

models = {
    'Linear Regression': linear_model,
    'Decision Tree Regression': decision_tree_model,
    'Random Forest Regression': random_forest_model,
    'Lasso Regression': lasso_model,
    'Bayesian Regression': bayesian_linear_model,
    'Stepwise Regression': stepwise_model  
}
print(models)
metrics = {
    'MSE': [linear_mse, decision_tree_mse, random_forest_mse, lasso_mse, bayesian_linear_mse, stepwise_mse],
    'RMSE': [linear_rmse, decision_tree_rmse, random_forest_rmse, lasso_rmse, bayesian_linear_rmse, stepwise_rmse],
    'MAE': [linear_mae, decision_tree_mae, random_forest_mae, lasso_mae, bayesian_linear_mae, stepwise_mae],
    'R-squared': [linear_r2, decision_tree_r2, random_forest_r2, lasso_r2, bayesian_linear_r2, stepwise_r2]
}


print(metrics)

# Print the evaluation metrics for each model
for model_name, model in models.items():
    print(f'{model_name}:')
    for metric_name, metric_values in metrics.items():
        print(f'{metric_name}: {metric_values[list(models.keys()).index(model_name)]:.2f}')
    print('-----------------------------')
    
# Choose the best model based on a specific evaluation metric
best_metric = 'RMSE'  # Choose the metric you want to use for model selection
best_model_name = min(models, key=lambda x: metrics[best_metric][list(models.keys()).index(x)])
print('-------------------------------')
print(f'Best Model (based on {best_metric}): {best_model_name}')

best_metric='MSE'
best_model_name=min(models,key=lambda x: metrics[best_metric][list(models.keys()).index(x)])
print('............................................')
print(f'Best Model(based on{best_metric}):{best_model_name}')

best_metric='MAE'
best_model_name=min(models,key=lambda x: metrics[best_metric][list(models.keys()).index(x)])
print('............................................')
print(f'Best Model(based on{best_metric}):{best_model_name}')
import matplotlib.pyplot as plt

# ... (previous code)

# Create a bar graph for RMSE
plt.figure(figsize=(10, 6))
plt.bar(models.keys(), metrics['RMSE'])
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) for Different Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

model_names = list(models.keys())
rmse_values = metrics['RMSE']

plt.figure(figsize=(10, 6))
plt.plot(model_names, rmse_values, marker='o', linestyle='-', color='b')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) for Different Models')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

x_values = [1, 2, 3, 4, 5]
y_values = [10, 15, 7, 12, 9]

import matplotlib.pyplot as plt

# List of model names
model_names = list(models.keys())

# List of MSE values
mse_values = metrics['MSE']

# Create a bar graph for MSE
plt.figure(figsize=(10, 6))
plt.bar(model_names, mse_values, color='blue')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('Mean Squared Error (MSE) for Different Models')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

import matplotlib.pyplot as plt

# List of model names
model_names = list(models.keys())

# List of MSE values
mse_values = metrics['MSE']

# Create a line plot for MSE
plt.figure(figsize=(10, 6))
plt.plot(model_names, mse_values, marker='o', linestyle='-', color='blue')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('Mean Squared Error (MSE) for Different Models')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()

import matplotlib.pyplot as plt

# List of model names
model_names = list(models.keys())

# List of MAE values
mae_values = metrics['MAE']

# Create a bar graph for MAE
plt.figure(figsize=(10, 6))
plt.bar(model_names, mae_values, color='green')
plt.xlabel('Models')
plt.ylabel('MAE')
plt.title('Mean Absolute Error (MAE) for Different Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# List of model names
model_names = list(models.keys())

# List of MAE values
mae_values = metrics['MAE']

# Create a line plot for MAE
plt.figure(figsize=(10, 6))
plt.plot(model_names, mae_values, marker='o', linestyle='-', color='green')
plt.xlabel('Models')
plt.ylabel('MAE')
plt.title('Mean Absolute Error (MAE) for Different Models')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# List of model names
model_names = list(models.keys())

# List of R-squared values
r2_values = metrics['R-squared']

# Create a bar graph for R-squared
plt.figure(figsize=(10, 6))
plt.bar(model_names, r2_values, color='purple')
plt.xlabel('Models')
plt.ylabel('R-squared')
plt.title('R-squared for Different Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# List of model names
model_names = list(models.keys())

# List of R-squared values
r2_values = metrics['R-squared']

# Create a line plot for R-squared
plt.figure(figsize=(10, 6))
plt.plot(model_names, r2_values, marker='o', linestyle='-', color='purple')
plt.xlabel('Models')
plt.ylabel('R-squared')
plt.title('R-squared for Different Models')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()








