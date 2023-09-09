Sunspot Data Analysis: A total of 133 data were collected from the international reference ionosphere over Guwahati city from the year 2012 to 2022 at distances 500, 1000, 1500 and 2000Km respectively. The analysis was between sunspot number vs electron density and electron temperature. Linear regression analysis has been used to analyse the data and get a appropriate corelation between them.

#Pseudo Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the figure size
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# Read the data from 'Data.csv' into a DataFrame
data = pd.read_csv('Data.csv')

# Extract the 'sunspot' and 'ne_500m^-3' columns as arrays
X = data['sunspot'].values
Y = data['ne_500m^-3'].values

# Calculate the mean of X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y) 


# Calculate the total number of values
m = len(X)

# Initialize the numerator and denominator for calculating the regression coefficients
numer = 0
denom = 0

# Calculate the numerator and denominator
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2

# Calculate the regression coefficients
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

# Print the regression coefficients
print(b1, b0)

# Set the maximum and minimum values for x
max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Generate evenly spaced values of x
x = np.linspace(min_x, max_x, 1000)

# Calculate the corresponding values of y based on the regression line equation
y = b0 + b1 * x

# Plot the regression line
plt.plot(x, y, color='red', label='Regression Line')

# Plot the scatter points
plt.scatter(X, Y, c='green', label='Scatter Plot')

# Set the x and y axis labels
plt.xlabel('sunspot')
plt.ylabel('ne')
 

# Add a legend
plt.legend()

# Show the plot
plt.show()
