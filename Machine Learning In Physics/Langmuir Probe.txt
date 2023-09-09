Langmuir Probe:  A Langmuir probe is a device that can determine the electron temperature, the electron density and the ion density of the plasma by obtaining its current-voltage characteristic since the characteristic curve depends on these plasma parameters. In this we have over 2000 experimental data divided into 5 datasets respectively. Polynomial regression is used to visualize the flow of the data points.	

#Pseudo Code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the data for CH1_Voltage and CH1_Current
X = df['CH1_Voltage'].values
Y = df['CH1_Current'].values

# Reshape the X data into a column matrix
X = X.reshape(-1, 1)

# Create PolynomialFeatures with degree 7
poly = PolynomialFeatures(degree=7)

# Transform the X data using polynomial features
X_poly = poly.fit_transform(X)

# Fit the polynomial features and target values to the LinearRegression model
poly.fit(X_poly, Y)
linreg = LinearRegression()
linreg.fit(X_poly, Y)
 

# Predict the target values using the fitted model
Y_pred = linreg.predict(X_poly)

# Plot the scatter points
plt.scatter(X, Y, color='blue')

# Plot the predicted values
plt.plot(X, Y_pred, color='red')

# Set the x and y axis labels
plt.xlabel('Voltage (V)')
plt.ylabel('Current (I)')

# Display the plot
plt.show()
