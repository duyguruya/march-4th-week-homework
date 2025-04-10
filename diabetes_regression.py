import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

diabetes_data = load_diabetes()

X = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
y = pd.Series(diabetes_data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_bmi_train = X_train[['BMI']]
X_bmi_test = X_test[['BMI']]

simple_model = LinearRegression()
simple_model.fit(X_bmi_train, y_train)

y_pred_simple = simple_model.predict(X_bmi_test)

r2_simple = r2_score(y_test, y_pred_simple)

mae_simple = mean_absolute_error(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)

multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)

y_pred_multiple = multiple_model.predict(X_test)

r2_multiple = r2_score(y_test, y_pred_multiple)

mae_multiple = mean_absolute_error(y_test, y_pred_multiple)
mse_multiple = mean_squared_error(y_test, y_pred_multiple)

print(f"Basit Model R²: {r2_simple}")
print(f"Çoklu Model R²: {r2_multiple}")
print(f"Basit Model MAE: {mae_simple}")
print(f"Çoklu Model MAE: {mae_multiple}")
print(f"Basit Model MSE: {mse_simple}")
print(f"Çoklu Model MSE: {mse_multiple}")
