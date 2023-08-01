import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from NObSP_Decomposition import NObSP_SVM_single, NObSP_SVM_2order
from sklearn.metrics.pairwise import linear_kernel,polynomial_kernel,rbf_kernel,laplacian_kernel,sigmoid_kernel,chi2_kernel
from sklearn.model_selection import train_test_split
from kernel_comp import kernel_comp

# Main script for the use pof NObSP in SVM. In this example I create 5 random variables, which are used to define
# 4 different nonlinear funcitons of a single variable, and one depending on the interaction between two variables
# One of the input has no relation with the output, so we expect its contribution to be zero. We created a dataset 
# wit N observations.
#
# @Copyrigth:  Alexander Caicedo, April 2023


N = 1000; # Defining the number of datapoints
kernel_used = 'rbf' # Selecting th ekernel to be used

# Input variables
x0 = np.random.randn(N,)/3
x1 = np.random.randn(N,)/3
x2 = np.random.randn(N,)/3
x3 = np.random.randn(N,)
x4 = np.random.randn(N,)/3

# Nonlinear functions definition
g0 = x0**2
g1 = x1**3
g2 = np.exp(x2)
g3 = np.sin(2*x3)
g4 = np.zeros(N,) # Notice that the contribution of x4 on the output is zero.
g01 = 1*np.exp((x0+x1)) # Second order interaction effect
noise = 0.01*np.random.randn(N,) # noise vector

# Generating the model output
y =  g0 + g1 + g2 + g3 + g4 + noise # Simulation done using only first order interactions in the output.
y2 =  g0 + g1 + g2 + g3 + g4 + g01 + noise # Simulation done using second order interactions in the output.

# Generating the input matriz for training
X = np.stack((x0, x1, x2, x3, x4), axis=1)
t = np.arange(0,N)

## Defining the Regression model for only the first order nonlinear effects

regressor_1 = SVR(kernel = kernel_used) # Creating the regression model
regressor_1.fit(X, y) # Fitting the model to the data
y_est_1=regressor_1.predict(X) # Prediction of the output in training data
Xsv_1 = regressor_1.support_vectors_# Extracting the support vectors 

# Decomposition using NObSP

P_xy, y_e_1, Alpha_1 = NObSP_SVM_single(X, Xsv_1, y_est_1, regressor_1)

K_1 = kernel_comp(X,Xsv_1,regressor_1) # computing the kernel for the training data
y_e_Alpha_1 = K_1@Alpha_1 # Obtaining the estimated decomposition using the Alpha coefficients, out-of-sample extension)

## Plotting the resutls

plt.plot(t,y,t,y_est_1) # Estimated output vs real output
plt.show()

# Plotting the decomposition
fig, axs = plt.subplots(2,3)
axs[0,0].scatter(x0,g0-g0.mean(), marker='x') # Real nonlienar contribution of x0 on the output
axs[0,0].scatter(x0,y_e_1[:,0]-y_e_1[:,0].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output

axs[0,1].scatter(x1,g1-g1.mean(), marker='x') # Real nonlienar contribution of x1 on the output
axs[0,1].scatter(x1,y_e_1[:,1]-y_e_1[:,1].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output

axs[0,2].scatter(x2,g2-g2.mean(), marker='x') # Real nonlienar contribution of x2 on the output
axs[0,2].scatter(x2,y_e_1[:,2]-y_e_1[:,2].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output

axs[1,0].scatter(x3,g3-g3.mean(), marker='x') # Real nonlienar contribution of x3 on the output
axs[1,0].scatter(x3,y_e_1[:,3]-y_e_1[:,3].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output

axs[1,1].scatter(x4,g4-g4.mean(), marker='x') # Real nonlienar contribution of x4 on the output
axs[1,1].scatter(x4,y_e_1[:,4]-y_e_1[:,4].mean(),marker='o') # Estimated nonlinear contribution of x04 on the output

axs[1,2].scatter(y,y_est_1, marker='x') # Calinbration plot real output vs estiamted output
plt.show()

# Plotting the decomposition using the out-of-sample extension
fig, axs = plt.subplots(2,3)
axs[0,0].scatter(x0,g0-g0.mean(), marker='x') # Real nonlienar contribution of x0 on the output
axs[0,0].scatter(x0,y_e_Alpha_1[:,0]-y_e_Alpha_1[:,0].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output

axs[0,1].scatter(x1,g1-g1.mean(), marker='x') # Real nonlienar contribution of x1 on the output
axs[0,1].scatter(x1,y_e_Alpha_1[:,1]-y_e_Alpha_1[:,1].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output

axs[0,2].scatter(x2,g2-g2.mean(), marker='x') # Real nonlienar contribution of x2 on the output
axs[0,2].scatter(x2,y_e_Alpha_1[:,2]-y_e_Alpha_1[:,2].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output

axs[1,0].scatter(x3,g3-g3.mean(), marker='x') # Real nonlienar contribution of x3 on the output
axs[1,0].scatter(x3,y_e_Alpha_1[:,3]-y_e_Alpha_1[:,3].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output

axs[1,1].scatter(x4,g4-g4.mean(), marker='x') # Real nonlienar contribution of x4 on the output
axs[1,1].scatter(x4,y_e_Alpha_1[:,4]-y_e_Alpha_1[:,4].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output

axs[1,2].scatter(y,y_est_1, marker='x') # Calinbration plot real output vs estiamted output
plt.show()

## Defining the Regression model with second order nonlinear interaction effects

regressor_2 = SVR(kernel = 'rbf') # Creating the regression model
regressor_2.fit(X, y2) # Fitting the model to the data
y_est_2=regressor_2.predict(X) # Prediction of the output in training data
Xsv_2 = regressor_2.support_vectors_# Extracting the support vectors 

P_xy_2, y_e_2, Alpha_2 = NObSP_SVM_single(X, Xsv_2, y_est_2, regressor_2)
P_xy_22, y_e_22 = NObSP_SVM_2order(X, Xsv_2, y_est_2, P_xy_2, regressor_2)

## Plotting the resutls

plt.plot(t,y2,t,y_est_2) # Estimated output vs real output
plt.show()
# Plotting the decomposition for first order interactions

fig, axs = plt.subplots(2,3)
axs[0,0].scatter(x0,g0-g0.mean(), marker='x') # Real nonlienar contribution of x0 on the output
axs[0,0].scatter(x0,y_e_2[:,0]-y_e_2[:,0].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output

axs[0,1].scatter(x1,g1-g1.mean(), marker='x') # Real nonlienar contribution of x1 on the output
axs[0,1].scatter(x1,y_e_2[:,1]-y_e_2[:,1].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output

axs[0,2].scatter(x2,g2-g2.mean(), marker='x') # Real nonlienar contribution of x2 on the output
axs[0,2].scatter(x2,y_e_2[:,2]-y_e_2[:,2].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output

axs[1,0].scatter(x3,g3-g3.mean(), marker='x') # Real nonlienar contribution of x3 on the output
axs[1,0].scatter(x3,y_e_2[:,3]-y_e_2[:,3].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output

axs[1,1].scatter(x4,g4-g4.mean(), marker='x') # Real nonlienar contribution of x4 on the output
axs[1,1].scatter(x4,y_e_2[:,4]-y_e_2[:,4].mean(),marker='o') # Estimated nonlinear contribution of x04 on the output

axs[1,2].scatter(y,y_est_2, marker='x') # Calinbration plot real output vs estiamted output
plt.show()

# Plottimng the second order interaction effects

fig, axs = plt.subplots(2,5)
axs[0,0].scatter((x0+x1), y_e_22[:,0]-y_e_22[:,0].mean(), marker='o') # Real interaction between x0 and x1
axs[0,0].scatter((x0+x1), g01-g01.mean(), marker='x') # Estimated interaction between x0 and x1

axs[0,1].plot(y_e_22[:,1]-y_e_22[:,1].mean()) # Estimated interaction between x0 and x2
axs[0,2].plot(y_e_22[:,2]-y_e_22[:,2].mean()) # Estimated interaction between x0 and x3
axs[0,3].plot(y_e_22[:,3]-y_e_22[:,3].mean()) # Estimated interaction between x0 and x4
axs[0,4].plot(y_e_22[:,4]-y_e_22[:,4].mean()) # Estimated interaction between x1 and x2
axs[1,0].plot(y_e_22[:,5]-y_e_22[:,5].mean()) # Estimated interaction between x1 and x3
axs[1,1].plot(y_e_22[:,6]-y_e_22[:,6].mean()) # Estimated interaction between x1 and x4
axs[1,2].plot(y_e_22[:,7]-y_e_22[:,7].mean()) # Estimated interaction between x2 and x3
axs[1,3].plot(y_e_22[:,8]-y_e_22[:,8].mean()) # Estimated interaction between x2 and x4
axs[1,4].plot(y_e_22[:,9]-y_e_22[:,9].mean()) # Estimated interaction between x3 and x4

plt.show()