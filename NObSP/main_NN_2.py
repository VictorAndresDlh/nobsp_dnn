import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from NObSP_Decomposition import ObSP, NObSP_SVM_single, NObSP_SVM_2order, NObSP_NN_single, NObSP_NN_2order, NObSP_NN_single_MultiOutput
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
epochs = 3000 # Definning numebr of epochs to train the models
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

g02 = np.cos(6*x0)
g12 = np.abs(x1)
g22 = np.tanh(6*x2)
g32 = np.zeros(N,)
g42 = np.zeros(N,)


noise = 0.01*np.random.randn(N,) # noise vector
noise2 = 0.01*np.random.randn(N,) # noise vector

# Generating the model output
y1 =  g0 + g1 + g2 + g3 + g4 + noise # Simulation done using only first order interactions in the output.
y2 =  g02 + g12 + g22 + g32 + g42 + noise2 # Simulation done using second order interactions in the output.

y = np.stack((y1, y2), axis=1)

# Generating the input matriz for training
X = np.stack((x0, x1, x2, x3, x4), axis=1)

g0= np.stack((g0, g02), axis=1)
g1= np.stack((g1, g12), axis=1)
g2= np.stack((g2, g22), axis=1)
g3= np.stack((g3, g32), axis=1)
g4= np.stack((g4, g42), axis=1)

t = np.arange(0,N)
p = np.size(y,1)
in_feat = np.size(X,1)

## Preparing the data for the model

train_split = int(0.8*N) # 80% of the data to be used as training

X = torch.from_numpy(X).type(torch.float) # Converting the input matrix to a Pytorch tensor format 
y = torch.from_numpy(y).type(torch.float) # Converting the output data to a Pytorch tensor format 

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

model_1 = Regression_NN(in_feat,p) # Creating the model
loss_fcn = nn.MSELoss() # Definning loss function
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=learning_rate)  # Defining optimizer

# Training loop for the model 1

for epoch in range(epochs):
    model_1.train() # Setting the model in training mode
    y_p, x_p = model_1(X_train) #forward pass
    
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

P_xy_1, y_e_1, Alpha_NN_1 = NObSP_NN_single_MultiOutput(X, y_est_1, model_1) # Computing the decomposition iusing NObSP. The Alpha parameters are the weigths for the Interpretation Layer

y_e_Alpha_1 = np.zeros((N,in_feat,p))
for i in range(p):
    y_e_Alpha_1[:,:,i] = (x_trans_total_1@Alpha_NN_1[:,i*in_feat:i*in_feat+in_feat]) # Computing the decomposition using the Alpha coefficients, out-of-sample extension

y_e_1 = y_e_1.cpu() # Setting th eoutput variables in the cpu.

## Plotting the resutls
for i in range(p):
    plt.plot(t,y[:,i],t,y_est_1[:,i]) # Estimated output vs real output
    plt.tight_layout()
    plt.show()


# Plotting the decomposition
for i in range(p):

    fig, axs = plt.subplots(2,3)
    plt.tight_layout()
    axs[0,0].scatter(x0,g0[:,i]-g0[:,i].mean(), marker='x') # Real nonlienar contribution of x0 on the output
    axs[0,0].scatter(x0,y_e_1[:,0,i]-y_e_1[:,0,i].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output

    axs[0,1].scatter(x1,g1[:,i]-g1[:,i].mean(), marker='x') # Real nonlienar contribution of x1 on the output
    axs[0,1].scatter(x1,y_e_1[:,1,i]-y_e_1[:,1,i].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output

    axs[0,2].scatter(x2,g2[:,i]-g2[:,i].mean(), marker='x') # Real nonlienar contribution of x2 on the output
    axs[0,2].scatter(x2,y_e_1[:,2,i]-y_e_1[:,2,i].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output

    axs[1,0].scatter(x3,g3[:,i]-g3[:,i].mean(), marker='x') # Real nonlienar contribution of x3 on the output
    axs[1,0].scatter(x3,y_e_1[:,3,i]-y_e_1[:,3,i].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output

    axs[1,1].scatter(x4,g4[:,i]-g4[:,i].mean(), marker='x') # Real nonlienar contribution of x4 on the output
    axs[1,1].scatter(x4,y_e_1[:,4,i]-y_e_1[:,4,i].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output

    axs[1,2].scatter(y[:,i].cpu(),y_est_1[:,i].cpu(), marker='x') # Calinbration plot real output vs estiamted output
    plt.show()

# Plotting the decomposition using the out-of-sample extension
for i in range(p):
    fig, axs = plt.subplots(2,3)
    plt.tight_layout()
    axs[0,0].scatter(x0,g0[:,i]-g0[:,i].mean(), marker='x') # Real nonlienar contribution of x0 on the output
    axs[0,0].scatter(x0,y_e_Alpha_1[:,0,i]-y_e_Alpha_1[:,0,i].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output

    axs[0,1].scatter(x1,g1[:,i]-g1[:,i].mean(), marker='x') # Real nonlienar contribution of x1 on the output
    axs[0,1].scatter(x1,y_e_Alpha_1[:,1,i]-y_e_Alpha_1[:,1,i].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output

    axs[0,2].scatter(x2,g2[:,i]-g2[:,i].mean(), marker='x') # Real nonlienar contribution of x2 on the output
    axs[0,2].scatter(x2,y_e_Alpha_1[:,2,i]-y_e_Alpha_1[:,2,i].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output

    axs[1,0].scatter(x3,g3[:,i]-g3[:,i].mean(), marker='x') # Real nonlienar contribution of x3 on the output
    axs[1,0].scatter(x3,y_e_Alpha_1[:,3,i]-y_e_Alpha_1[:,3,i].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output

    axs[1,1].scatter(x4,g4[:,i]-g4[:,i].mean(), marker='x') # Real nonlienar contribution of x4 on the output
    axs[1,1].scatter(x4,y_e_Alpha_1[:,4,i]-y_e_Alpha_1[:,4,i].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output

    axs[1,2].scatter(y.cpu(),y_est_1.cpu(), marker='x') # Calinbration plot real output vs estiamted output
    plt.show()

## Estimating the parameters for the output layer in the modified arquitecture

Alpha_out_layer = torch.zeros(in_feat*p,p).type(torch.float)
b_out_layer = torch.zeros(1,p)

for i in range(p):
    X_final_1 = torch.cat((torch.from_numpy(y_e_Alpha_1[:,:,i]).squeeze(), torch.ones(N,1)),dim=1) # Extending the matrix of the estimated contributions with a vector of ones to find te value of the bias term

    # Solving the least squares problem between the output of the interpretation layer, and the real output. Ideally 
    # the weigths for this layer should be all 1. However, for numerical errors and to correct for a possible deviation by an scalar 
    # the least square problem is solved.

    Sol = torch.linalg.lstsq(X_final_1.type(torch.float),y[:,i], rcond=None, driver='gelsd')[0] 
    Alpha_out_layer[i*in_feat:i*in_feat+in_feat,i] = torch.t(Sol[:-1]) # Extracting the weigths for the output layer
    b_out_layer[0,i]= Sol[-1] # Extracting the bias for the output layer

print(Alpha_out_layer)

# Creating the model with the interpretable layer. This model uses the model where the data was trained, but it adds an 
# Interpretable layer between the last hidden layer and the output layer of the model. The weigths fo rthe interpretable layer 
# are the coefficients Alpha_NN_1, the bias term are set to 0. The last layer has as weigths the parameters Alpha_out_layer 
# and its bias term b_out_layer

model_1_Inter = Regression_NN_NObSP(model_1, torch.t(Alpha_NN_1), torch.t(Alpha_out_layer), b_out_layer,p) # Creating the Interpretable model

# Evalauting the model   
model_1_Inter.eval() # Setting th emodel in evaluation mode
with torch.inference_mode():
    y_est_1_Inter, y_est_1_Inter_dec = model_1_Inter(X) # Computing th eoutput of the Interpretable model, the estimated final output and the decomposition

# Plotting the results for the estimated output
for i in range(p):
    
    fig, axs = plt.subplots(4,1)
    plt.tight_layout()
    axs[0].plot(t, y_est_1_Inter[:,i].cpu(), t, y_est_1[:,i].cpu())  # Estimated output of the interpretable model vs the original model
    axs[1].scatter(y_est_1[:,i].cpu(), y_est_1_Inter[:,i].cpu(), marker='x')  # Calibration plot between estimated output of the original model vs estimated output of the Interpretable model
    axs[2].plot(t, y[:,i], t, y_est_1_Inter[:,i].cpu()) # Estimated output of the interpretable model vs real output
    axs[3].scatter(y[:,i], y_est_1_Inter[:,i].cpu(), marker='x') # Calibration plot between estimated output of the Interpretable model vs real output
    plt.show()

# Plotting the results for the estimated decomposition real and using the model
for i in range(p):
    
    fig, axs = plt.subplots(2,3)
    plt.tight_layout()
    axs[0,0].scatter(x0,g0[:,i]-g0[:,i].mean(), marker='x') # Real nonlienar contribution of x0 on the output
    axs[0,0].scatter(x0,y_est_1_Inter_dec[:,i*in_feat+0]-y_est_1_Inter_dec[:,i*in_feat+0].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output

    axs[0,1].scatter(x1,g1[:,i]-g1[:,i].mean(), marker='x') # Real nonlienar contribution of x1 on the output
    axs[0,1].scatter(x1,y_est_1_Inter_dec[:,i*in_feat+1]-y_est_1_Inter_dec[:,i*in_feat+1].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output

    axs[0,2].scatter(x2,g2[:,i]-g2[:,i].mean(), marker='x') # Real nonlienar contribution of x2 on the output
    axs[0,2].scatter(x2,y_est_1_Inter_dec[:,i*in_feat+2]-y_est_1_Inter_dec[:,i*in_feat+2].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output

    axs[1,0].scatter(x3,g3[:,i]-g3[:,i].mean(), marker='x') # Real nonlienar contribution of x3 on the output
    axs[1,0].scatter(x3,y_est_1_Inter_dec[:,i*in_feat+3]-y_est_1_Inter_dec[:,i*in_feat+3].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output

    axs[1,1].scatter(x4,g4[:,i]-g4[:,i].mean(), marker='x') # Real nonlienar contribution of x4 on the output
    axs[1,1].scatter(x4,y_est_1_Inter_dec[:,i*in_feat+4]-y_est_1_Inter_dec[:,i*in_feat+4].mean(),marker='o') # Estimated nonlinear contribution of x4 on the output

    axs[1,2].scatter(y[:,i].cpu(),y_est_1_Inter[:,i].cpu(), marker='x') # Calinbration plot real output vs estiamted output
    plt.show()

# Estimation error in the projections

error_Model = torch.zeros((N,p))
error_Model_Approx = torch.zeros((N,p))
error_Approx = torch.zeros((N,p))

for i in range(p):
    error_Model[:,i] = y[:,i]-y_est_1[:,i]
    error_Model_Approx[:,i] = y[:,i]-y_est_1_Inter[:,i]
    error_Approx [:,i]= y_est_1[:,i]-y_est_1_Inter[:,i]

# Plotting the errors
for i in range(p):
    
    fig, axs = plt.subplots(3,1)
    axs[0].plot(error_Model[:,i])
    axs[1].plot(error_Model_Approx[:,i])
    axs[2].plot(error_Approx[:,i])
    plt.show()

Mag_error_Model = np.zeros((p,1))
Mag_error_Model_Approx = np.zeros((p,1))
Mag_error_Approx = np.zeros((p,1))

for i in range(p):
    
    Mag_error_Model[i] = torch.dot(error_Model[:,i].squeeze(),error_Model[:,i].squeeze())/N
    Mag_error_Model_Approx[i] = torch.dot(error_Model_Approx[:,i].squeeze(),error_Model_Approx[:,i].squeeze())/N
    Mag_error_Approx[i] = torch.dot(error_Approx[:,i].squeeze(),error_Approx[:,i].squeeze())/N

    print(f'The error in the prediction of the original model for the output y{i} is: {Mag_error_Model[i]}')
    print(f'The error in the prediction of the interpretable model for the output y{i} is: {Mag_error_Model_Approx[i]}')
    print(f'The error between the predictions of the original and the interpretable model for the output y{i} is: {Mag_error_Approx[i]}')