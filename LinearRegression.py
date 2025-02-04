from sklearn.datasets import fetch_openml

boston = fetch_openml(name="boston", version=1, as_frame=True)
# Return data in Pandas DataFrame format (otherwise it will be a NumPy array).
data = boston.data
target = boston.target
print(data.columns)

df = data[['RM']].copy()  
df['Price'] = target  

# Remove missing values
df = df.dropna()

X = df[['RM']] # Use df[['RM']] to keep X in a 2D format (n_samples, n_features)
y = df[['Price']]

print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# 20% data as test set, 80% data for training

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 
# Make predictions on the test set, calculate the estimated house prices
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

import numpy as np # Generate smooth X-axis data
import matplotlib.pyplot as plt

X_range = np.linspace(X_test['RM'].min(), X_test['RM'].max(), 100).reshape(-1, 1) 
# Generate 100 equally spaced X values (for plotting the smooth regression line), .reshape(-1, 1) converts the array to 2D format (n_samples, 1)

y_pred_smooth = model.predict(X_range) 
# Calculate the predicted values on the regression line

plt.scatter(X_test['RM'], y_test, color='blue', label="True Values")
plt.plot(X_range, y_pred_smooth, color='red', linewidth=2, label="Predicted Regression Line")

plt.xlabel("Room Numbers")
plt.ylabel("Price")
plt.legend()
plt.show()
