import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from NObSP_Decomposition import ObSP, NObSP_SVM_single, NObSP_SVM_2order, NObSP_NN_single, NObSP_NN_2order
from sklearn.model_selection import train_test_split
from Regression_NN import Regression_NN
from Regression_NN_NObSP import Regression_NN_NObSP

# Main script for the use pof NObSP in NN. In this example I create 5 random variables, which are used to define
# 4 different nonlinear funcitons of a single variable, and one depending on the interaction between two variables
# One of the input has no relation with the output, so we expect its contribution to be zero. We created a dataset 
# wit N observations.
#
# @Copyrigth:  Alexander Caicedo, April 2023


N = 1000; # Defining the number of datapoints
epochs = 2000 # Definning numebr of epochs to train the models
learning_rate = 0.05 # DEfining learning rate of the model

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

## Preparing the data for the model

train_split = int(0.8*N) # 80% of the data to be used as training

X = torch.from_numpy(X).type(torch.float) # Converting the input matrix to a Pytorch tensor format 
y = torch.from_numpy(y).type(torch.float).unsqueeze(dim=-1) # Converting the output data to a Pytorch tensor format 

# Normalizing the input data
X_mean = torch.mean(X,dim=0) 
X_var = torch.var(X,dim=0)
X = (X-X_mean)/X_var

# Split of the data for training and test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    train_size = train_split,
                                                    random_state=42) 


## Creating model for single nonlinear effects

model_1 = Regression_NN(5,1) # Creating the model
loss_fcn = nn.MSELoss() # Definning loss function
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=learning_rate)  # Defining optimizer

# Training loop for the model 1

for epoch in range(epochs):
    model_1.train() # Setting the model in training mode
    y_p, x_p = model_1(X_train) #forward pass
    #y_p.squeeze()
    loss = loss_fcn(y_p,
                    y_train)  # Compute Loss
    loss.backward() # compute backward
    optimizer.step() # update parameters
    optimizer.zero_grad() #zero grad optimizer
    
    ## Testing
    model_1.eval() # Setting the model in evalaution mode
    with torch.inference_mode():
        y_pred, x_trans = model_1(X_test) # Estimating th emodel output in test data
    
    test_loss = loss_fcn(y_pred,
                         y_test) # Evaluating loss
    
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.5f} | test Loss: {test_loss:.5f}') # Printing the performance of the model as it is trained
    

model_1.eval() # Setting the model in evaluation mode
with torch.inference_mode():
    y_est_1, x_trans_total_1 = model_1(X) # Computin ght enonlinear transformation of the input data X

P_xy_1, y_e_1, Alpha_NN_1 = NObSP_NN_single(X, y_est_1, model_1) # Computing the decomposition iusing NObSP. The Alpha parameters are the weigths for the Interpretation Layer
y_e_Alpha_1 = x_trans_total_1@Alpha_NN_1 # Computing the decomposition using the Alpha coefficients, out-of-sample extension

y_e_1 = y_e_1.cpu() # Setting th eoutput variables in the cpu.

## Plotting the resutls
plt.plot(t,y,t,y_est_1) # Estimated output vs real output
plt.tight_layout()
plt.show()

# Plotting the decomposition
fig, axs = plt.subplots(2,3)
plt.tight_layout()
axs[0,0].scatter(x0,g0-g0.mean(), marker='x') # Real nonlienar contribution of x0 on the output
axs[0,0].scatter(x0,y_e_1[:,0]-y_e_1[:,0].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output

axs[0,1].scatter(x1,g1-g1.mean(), marker='x') # Real nonlienar contribution of x1 on the output
axs[0,1].scatter(x1,y_e_1[:,1]-y_e_1[:,1].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output

axs[0,2].scatter(x2,g2-g2.mean(), marker='x') # Real nonlienar contribution of x2 on the output
axs[0,2].scatter(x2,y_e_1[:,2]-y_e_1[:,2].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output

axs[1,0].scatter(x3,g3-g3.mean(), marker='x') # Real nonlienar contribution of x3 on the output
axs[1,0].scatter(x3,y_e_1[:,3]-y_e_1[:,3].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output

axs[1,1].scatter(x4,g4-g4.mean(), marker='x') # Real nonlienar contribution of x4 on the output
axs[1,1].scatter(x4,y_e_1[:,4]-y_e_1[:,4].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output

axs[1,2].scatter(y.cpu(),y_est_1.cpu(), marker='x') # Calinbration plot real output vs estiamted output
plt.show()

# Plotting the decomposition using the out-of-sample extension
fig, axs = plt.subplots(2,3)
plt.tight_layout()
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

axs[1,2].scatter(y.cpu(),y_est_1.cpu(), marker='x') # Calinbration plot real output vs estiamted output
plt.show()

## Estimating the parameters for the output layer in the modified arquitecture

X_final_1 = torch.cat((y_e_Alpha_1, torch.ones(N,1)),dim=1) # Extending the matrix of the estimated contributions with a vector of ones to find te value of the bias term

# Solving the least squares problem between the output of the interpretation layer, and the real output. Ideally 
# the weigths for this layer should be all 1. However, for numerical errors and to correct for a possible deviation by an scalar 
# the least square problem is solved.

Sol = torch.linalg.lstsq(X_final_1,y, rcond=None, driver='gelsd')[0] 
Alpha_out_layer = torch.t(Sol[:-1]) # Extracting the weigths for the output layer
b_out_layer= Sol[-1] # Extracting the bias for the output layer

# Creating the model with the interpretable layer. This model uses the model where the data was trained, but it adds an 
# Interpretable layer between the last hidden layer and the output layer of the model. The weigths fo rthe interpretable layer 
# are the coefficients Alpha_NN_1, the bias term are set to 0. The last layer has as weigths the parameters Alpha_out_layer 
# and its bias term b_out_layer

model_1_Inter = Regression_NN_NObSP(model_1, torch.t(Alpha_NN_1), Alpha_out_layer, b_out_layer,1) # Creating the Interpretable model

# Evalauting the model   
model_1_Inter.eval() # Setting th emodel in evaluation mode
with torch.inference_mode():
    y_est_1_Inter, y_est_1_Inter_dec = model_1_Inter(X) # Computing th eoutput of the Interpretable model, the estimated final output and the decomposition

# Plotting the results for the estimated output
fig, axs = plt.subplots(4,1)
plt.tight_layout()
axs[0].plot(t, y_est_1_Inter.cpu(), t, y_est_1.cpu())  # Estimated output of the interpretable model vs the original model
axs[1].scatter(y_est_1.cpu(), y_est_1_Inter.cpu(), marker='x')  # Calibration plot between estimated output of the original model vs estimated output of the Interpretable model
axs[2].plot(t, y, t, y_est_1_Inter.cpu()) # Estimated output of the interpretable model vs real output
axs[3].scatter(y, y_est_1_Inter.cpu(), marker='x') # Calibration plot between estimated output of the Interpretable model vs real output
plt.show()

# Plotting the results for the estimated decomposition real and using the model
fig, axs = plt.subplots(2,3)
plt.tight_layout()
axs[0,0].scatter(x0,g0-g0.mean(), marker='x') # Real nonlienar contribution of x0 on the output
axs[0,0].scatter(x0,y_est_1_Inter_dec[:,0]-y_est_1_Inter_dec[:,0].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output

axs[0,1].scatter(x1,g1-g1.mean(), marker='x') # Real nonlienar contribution of x1 on the output
axs[0,1].scatter(x1,y_est_1_Inter_dec[:,1]-y_est_1_Inter_dec[:,1].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output

axs[0,2].scatter(x2,g2-g2.mean(), marker='x') # Real nonlienar contribution of x2 on the output
axs[0,2].scatter(x2,y_est_1_Inter_dec[:,2]-y_est_1_Inter_dec[:,2].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output

axs[1,0].scatter(x3,g3-g3.mean(), marker='x') # Real nonlienar contribution of x3 on the output
axs[1,0].scatter(x3,y_est_1_Inter_dec[:,3]-y_est_1_Inter_dec[:,3].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output

axs[1,1].scatter(x4,g4-g4.mean(), marker='x') # Real nonlienar contribution of x4 on the output
axs[1,1].scatter(x4,y_est_1_Inter_dec[:,4]-y_est_1_Inter_dec[:,4].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output

axs[1,2].scatter(y.cpu(),y_est_1_Inter.cpu(), marker='x') # Calinbration plot real output vs estiamted output
plt.show()

# Estimation error in the projections

error_Model = y-y_est_1;
error_Model_Approx = y-y_est_1_Inter;
error_Approx = y_est_1-y_est_1_Inter;

# Plotting the errors
fig, axs = plt.subplots(3,1)
axs[0].plot(error_Model)
axs[1].plot(error_Model_Approx)
axs[2].plot(error_Approx)
plt.show()

Mag_error_Model = torch.dot(error_Model.squeeze(),error_Model.squeeze())/N
Mag_error_Model_Approx = torch.dot(error_Model_Approx.squeeze(),error_Model_Approx.squeeze())/N
Mag_error_Approx = torch.dot(error_Approx.squeeze(),error_Approx.squeeze())/N

print(f'The error in the prediction of the original model is: {Mag_error_Model}')
print(f'The error in the prediction of the interpretable model is: {Mag_error_Model_Approx}')
print(f'The error between the predictions of the original and the interpretable model is: {Mag_error_Approx}')