•	Electron and Plasma interaction phenomenon: We have applied decision tree in the beam speed and the percent retrieval of kinetic energy dataset to predict the missing values and also plot the actual values with the predicted values.

#Pseudo Code
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Open a file to write the results
file = open("file.txt", "w")

# Read the data from 'vd.csv' into a DataFrame
vd_data = pd.read_csv('vd.csv')

# Split the data into input features (X) and target variable (Y)
X = vd_data.drop(columns=['value'])
Y = vd_data['value']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Create a DecisionTreeClassifier model
model = DecisionTreeClassifier()

# Fit the model to the training data
model.fit(X_train.values, Y_train) 


# Ask for user input
n = int(input('Enter vd: '))

# Make a prediction for the input value
predictions = model.predict([[n]])
print(predictions)

# Write the predictions to the file for values from 0 to 99
for i in range(100):
    predictions = model.predict([[i]])
    file.write(str(i) + '\t' + str(predictions[0]) + '\n')

# Load the data from the file
x, y = np.loadtxt("file.txt", unpack=True)
T, Z = np.loadtxt("vd.txt", unpack=True)

# Plot the data
plt.plot(x, y, 'r.-', label='Predicted')
plt.plot(T, Z, 'b.-', label='Actual')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()

Then we apply both linear and polynomial regression in this dataset and we observe that polynomial regression solves the problem which was occurring due to linear regression. We get the best fit line using polynomial regression.

# Pseudo Code
import matplotlib.pyplot as plt
import matplotlib as mp
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the data from 'ke_analysis_data.txt'
x, y = np.loadtxt('ke_analysis_data.txt', unpack=True)

# Reshape the x-array into a column matrix 

x = x.reshape([len(x), 1])

# Create a LinearRegression model and fit it to the data
lin = LinearRegression()
lin.fit(x, y)

# Create PolynomialFeatures with degree 6 and transform the x data
poly = PolynomialFeatures(degree=6)
x_poly = poly.fit_transform(x)

# Fit the polynomial features and target values to the PolynomialRegression model
poly.fit(x_poly, y)
lin2 = LinearRegression()
lin2.fit(x_poly, y)

# Set matplotlib configurations
mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)

# Create a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Plot the data and linear regression line in the first subplot
ax[0].plot(x, y, 'r.-', markersize=20)
ax[0].plot(x, lin.predict(x), color='blue')
ax[0].set_title('Linear Regression')
ax[0].set_xlabel('$v_{d}$')
ax[0].set_ylabel('$\Delta$ $KE_{B}$(%)')

# Plot the data and polynomial regression line in the second subplot
ax[1].plot(x, y, 'r.-', markersize=20)
ax[1].plot(x, lin2.predict(poly.fit_transform(x)), color='green')
ax[1].set_title('Polynomial Regression')
ax[1].set_xlabel('$v_{d}$')
ax[1].set_ylabel('$\Delta$ $KE_{B}$(%)')

# Display the plot 

plt.show()

# Predict a new result using Linear Regression
pred = 110.0
predarray = np.array([[pred]])
lin.predict(predarray)

# Predict a new result using Polynomial Regression
pred2 = 110.0
pred2array = np.array([[pred2]])
lin2.predict(poly.fit_transform(pred2array))
