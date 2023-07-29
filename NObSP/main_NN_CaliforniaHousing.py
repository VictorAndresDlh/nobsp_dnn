import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from NObSP_Decomposition import ObSP, NObSP_SVM_single, NObSP_SVM_2order, NObSP_NN_single, NObSP_NN_2order
from sklearn.model_selection import train_test_split
from Regression_NN import Regression_NN
from Regression_NN_NObSP import Regression_NN_NObSP
from sklearn.datasets import fetch_california_housing


# Loading the data
housing = fetch_california_housing()
X = housing.data
y = housing.target
print(housing)

# Initializing variables
epochs = 300
learning_rate = 0.05 # DEfining learning rate of the model
N = len(y)
N1 = 6000 # number of samples to train and compute projections
train_split = int(0.8*N1) # 80% of the data to be used as training and compute projections

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

x0 = X_train[:,0]
x1 = X_train[:,1]
x2 = X_train[:,2]
x3 = X_train[:,3]
x4 = X_train[:,4]
x5 = X_train[:,5]
x6 = X_train[:,6]
x7 = X_train[:,7]

## Creating model for single nonlinear effects

model_1 = Regression_NN(8,1) # Creating the model
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
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.5f} | test Loss: {test_loss:.5f}') # Printing the performance of the model as it is trained
    

model_1.eval() # Setting the model in evaluation mode
with torch.inference_mode():
    y_est_1, x_trans_total_1 = model_1(X_train) # Computin ght enonlinear transformation of the input data X

P_xy_1, y_e_1, Alpha_NN_1 = NObSP_NN_single(X_train, y_est_1, model_1) # Computing the decomposition iusing NObSP. The Alpha parameters are the weigths for the Interpretation Layer
y_e_Alpha_1 = x_trans_total_1@Alpha_NN_1 # Computing the decomposition using the Alpha coefficients, out-of-sample extension

y_e_1 = y_e_1.cpu() # Setting th eoutput variables in the cpu.

t = np.arange(0,train_split)
## Plotting the resutls
plt.plot(t,y_train,t,y_est_1) # Estimated output vs real output
plt.show()

# Plotting the decomposition
fig, axs = plt.subplots(3,4)
axs[0,0].scatter(x0,y_e_1[:,0]-y_e_1[:,0].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output
axs[0,1].scatter(x1,y_e_1[:,1]-y_e_1[:,1].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output
axs[0,2].scatter(x2,y_e_1[:,2]-y_e_1[:,2].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output
axs[0,3].scatter(x3,y_e_1[:,3]-y_e_1[:,3].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output
axs[1,0].scatter(x4,y_e_1[:,4]-y_e_1[:,4].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[1,1].scatter(x5,y_e_1[:,5]-y_e_1[:,5].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[1,2].scatter(x6,y_e_1[:,6]-y_e_1[:,6].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[1,3].scatter(x7,y_e_1[:,7]-y_e_1[:,7].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[2,1].scatter(y_train.cpu(),y_est_1.cpu(), marker='x') # Calinbration plot real output vs estiamted output
plt.show()

# Plotting the decomposition using the out-of-sample extension
fig, axs = plt.subplots(3,4)
axs[0,0].scatter(x0,y_e_Alpha_1[:,0]-y_e_Alpha_1[:,0].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output
axs[0,1].scatter(x1,y_e_Alpha_1[:,1]-y_e_Alpha_1[:,1].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output
axs[0,2].scatter(x2,y_e_Alpha_1[:,2]-y_e_Alpha_1[:,2].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output
axs[0,3].scatter(x3,y_e_Alpha_1[:,3]-y_e_Alpha_1[:,3].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output
axs[1,0].scatter(x4,y_e_Alpha_1[:,4]-y_e_Alpha_1[:,4].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[1,1].scatter(x5,y_e_Alpha_1[:,5]-y_e_Alpha_1[:,5].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[1,2].scatter(x6,y_e_Alpha_1[:,6]-y_e_Alpha_1[:,6].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[1,3].scatter(x7,y_e_Alpha_1[:,7]-y_e_Alpha_1[:,7].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[2,1].scatter(y_train.cpu(),y_est_1.cpu(), marker='x') # Calinbration plot real output vs estiamted output
plt.show()

## Estimating the parameters for the output layer in the modified arquitecture

X_final_1 = torch.cat((y_e_Alpha_1, torch.ones(train_split,1)),dim=1) # Extending the matrix of the estimated contributions with a vector of ones to find te value of the bias term

# Solving the least squares problem between the output of the interpretation layer, and the real output. Ideally 
# the weigths for this layer should be all 1. However, for numerical errors and to correct for a possible deviation by an scalar 
# the least square problem is solved.

Sol = torch.linalg.lstsq(X_final_1,y_train, rcond=None, driver='gelsd')[0] 
Alpha_out_layer = torch.t(Sol[:-1]) # Extracting the weigths for the output layer
b_out_layer= Sol[-1] # Extracting the bias for the output layer

# Creating the model with the interpretable layer. This model uses the model where the data was trained, but it adds an 
# Interpretable layer between the last hidden layer and the output layer of the model. The weigths fo rthe interpretable layer 
# are the coefficients Alpha_NN_1, the bias term are set to 0. The last layer has as weigths the parameters Alpha_out_layer 
# and its bias term b_out_layer

model_1_Inter = Regression_NN_NObSP(model_1, torch.t(Alpha_NN_1), Alpha_out_layer, b_out_layer,1) # Creating the Interpretable model

# Evalauting the model   
model_1_Inter.eval() # Setting th emodel in evaluation mode
model_1.eval() # Setting th emodel in evaluation mode
with torch.inference_mode():
    y_est_1_Inter, y_est_1_Inter_dec = model_1_Inter(X) # Computing th eoutput of the Interpretable model, the estimated final output and the decomposition
    y_est_total, Intermediate = model_1(X)
    
t = np.arange(0,N)
# Plotting the results for the estimated output
fig, axs = plt.subplots(4,1)
axs[0].plot(t, y, t, y_est_total.cpu()) # Estimated output of the complete model vs real output
axs[1].scatter(y, y_est_total.cpu(), marker='x') # Calibration plot between estimated output of the complete model vs real output
axs[2].plot(t, y, t, y_est_1_Inter.cpu()) # Estimated output of the interpretable model vs real output
axs[3].scatter(y, y_est_1_Inter.cpu(), marker='x') # Calibration plot between estimated output of the Interpretable model vs real output
plt.show()

x0 = X[:,0]
x1 = X[:,1]
x2 = X[:,2]
x3 = X[:,3]
x4 = X[:,4]
x5 = X[:,5]
x6 = X[:,6]
x7 = X[:,7]

# Plotting the results for the estimated decomposition real and using the model
fig, axs = plt.subplots(3,4)
axs[0,0].scatter(x0,y_est_1_Inter_dec[:,0]-y_est_1_Inter_dec[:,0].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output
axs[0,1].scatter(x1,y_est_1_Inter_dec[:,1]-y_est_1_Inter_dec[:,1].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output
axs[0,2].scatter(x2,y_est_1_Inter_dec[:,2]-y_est_1_Inter_dec[:,2].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output
axs[0,3].scatter(x3,y_est_1_Inter_dec[:,3]-y_est_1_Inter_dec[:,3].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output
axs[1,0].scatter(x4,y_est_1_Inter_dec[:,4]-y_est_1_Inter_dec[:,4].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[1,1].scatter(x5,y_est_1_Inter_dec[:,5]-y_est_1_Inter_dec[:,5].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[1,2].scatter(x6,y_est_1_Inter_dec[:,6]-y_est_1_Inter_dec[:,6].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[1,3].scatter(x7,y_est_1_Inter_dec[:,7]-y_est_1_Inter_dec[:,7].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output
axs[2,1].scatter(y.cpu(),y_est_1_Inter.cpu(), marker='x') # Calinbration plot real output vs estiamted output
plt.show()

# Estimation error in the projections

error_Model = y-y_est_total;
error_Model_Approx = y-y_est_1_Inter;
error_Approx = y_est_total-y_est_1_Inter;

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