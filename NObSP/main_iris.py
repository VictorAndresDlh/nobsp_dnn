import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from NObSP_Decomposition import ObSP, NObSP_SVM_single, NObSP_SVM_2order, NObSP_NN_single, NObSP_NN_2order, NObSP_NN_single_MultiOutput, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from Classifier_NN import Classifier_NN
from Classifier_NN_NObSP import Classifier_NN_NObSP
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from sklearn.neighbors import KernelDensity

# Main script for the use pof NObSP in NN. In this example I create a classifier for th eiris datast and 
# attemp to decpmpose its output for interpretbility
#
# @Copyrigth:  Alexander Caicedo, April 2023

epochs = 6000 # Definning numebr of epochs to train the models
learning_rate = 0.02 # Defining learning rate of the model

# Loading and preparing input
iris = load_iris()
print(iris)
X = iris['data']
y_in = iris['target']
names = iris['target_names']
p = len(np.unique(y_in))
y_int = y_in.astype(np.int32)
y = to_categorical(y_in, p)
N = np.size(X,0); # Defining the number of datapoints
in_feat = np.size(X,1)

# Input variables
x0 = X[:,0]
x1 = X[:,1]
x2 = X[:,2]
x3 = X[:,3]

t = np.arange(0,N)
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

model_1 = Classifier_NN(in_feat,p) # Creating the model
loss_fcn = nn.CrossEntropyLoss() # Definning loss function
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=learning_rate)  # Defining optimizer

# Training loop for the model 1

for epoch in range(epochs):
    model_1.train() # Setting the model in training mode
    y_prob, x_p, y_lin = model_1(X_train) #forward pass
    
    loss = loss_fcn(y_prob,
                    y_train)  # Compute Loss
    loss.backward() # compute backward
    optimizer.step() # update parameters
    optimizer.zero_grad() #zero grad optimizer
    
    ## Testing
    model_1.eval() # Setting the model in evalaution mode
    with torch.inference_mode():
        y_prob_pred, x_trans, y_lin_pred = model_1(X_test) # Estimating th emodel output in test data
    
    test_loss = loss_fcn(y_prob_pred,
                         y_test) # Evaluating loss
    
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.5f} | test Loss: {test_loss:.5f}') # Printing the performance of the model as it is trained
    
model_1.eval() # Setting the model in evaluation mode
with torch.inference_mode():
    y_prob_1, x_trans_total_1, y_est_1 = model_1(X) # Computin ght enonlinear transformation of the input data X

P_xy_1, y_e_1, Alpha_NN_1 = NObSP_NN_single_MultiOutput(X, y_est_1, model_1) # Computing the decomposition iusing NObSP. The Alpha parameters are the weigths for the Interpretation Layer

y_e_Alpha_1 = np.zeros((N,in_feat,p))
for i in range(p):
    y_e_Alpha_1[:,:,i] = (x_trans_total_1@Alpha_NN_1[:,i*in_feat:i*in_feat+in_feat]) # Computing the decomposition using the Alpha coefficients, out-of-sample extension

y_e_1 = y_e_1.cpu() # Setting th eoutput variables in the cpu.

## Plotting the resutls
for i in range(p):
    plt.plot(t,y[:,i],t,y_prob_1[:,i]) # Estimated output vs real output
    plt.tight_layout()
    #plt.show()

# Plotting the decomposition
for i in range(p):

    fig, axs = plt.subplots(2,3)
    plt.tight_layout()
    axs[0,0].scatter(x0,y_e_1[:,0,i]-y_e_1[:,0,i].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output
    axs[0,1].scatter(x1,y_e_1[:,1,i]-y_e_1[:,1,i].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output
    axs[0,2].scatter(x2,y_e_1[:,2,i]-y_e_1[:,2,i].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output
    axs[1,0].scatter(x3,y_e_1[:,3,i]-y_e_1[:,3,i].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output
    axs[1,1].scatter(y[:,i].cpu(),y_prob_1[:,i].cpu(), marker='x') # Calinbration plot real output vs estiamted output
    #plt.show()

# Plotting the decomposition using the out-of-sample extension
for i in range(p):
    fig, axs = plt.subplots(2,3)
    plt.tight_layout()
    axs[0,0].scatter(x0,y_e_Alpha_1[:,0,i]-y_e_Alpha_1[:,0,i].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output
    axs[0,1].scatter(x1,y_e_Alpha_1[:,1,i]-y_e_Alpha_1[:,1,i].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output
    axs[0,2].scatter(x2,y_e_Alpha_1[:,2,i]-y_e_Alpha_1[:,2,i].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output
    axs[1,0].scatter(x3,y_e_Alpha_1[:,3,i]-y_e_Alpha_1[:,3,i].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output
    axs[1,1].scatter(y.cpu(),y_prob_1.cpu(), marker='x') # Calinbration plot real output vs estiamted output
    #plt.show()

## Estimating the parameters for the output layer in the modified arquitecture

Alpha_out_layer = torch.zeros(in_feat*p,p).type(torch.float)
b_out_layer = torch.zeros(1,p)

for i in range(p):
    X_final_1 = torch.cat((torch.from_numpy(y_e_Alpha_1[:,:,i]).squeeze(), torch.ones(N,1)),dim=1) # Extending the matrix of the estimated contributions with a vector of ones to find te value of the bias term

    # Solving the least squares problem between the output of the interpretation layer, and the real output. Ideally 
    # the weigths for this layer should be all 1. However, for numerical errors and to correct for a possible deviation by an scalar 
    # the least square problem is solved.

    Sol = torch.linalg.lstsq(X_final_1.type(torch.float),y_est_1[:,i], rcond=None, driver='gelsd')[0] 
    Alpha_out_layer[i*in_feat:i*in_feat+in_feat,i] = torch.t(Sol[:-1]) # Extracting the weigths for the output layer
    b_out_layer[0,i]= Sol[-1] # Extracting the bias for the output layer

# Creating the model with the interpretable layer. This model uses the model where the data was trained, but it adds an 
# Interpretable layer between the last hidden layer and the output layer of the model. The weigths fo rthe interpretable layer 
# are the coefficients Alpha_NN_1, the bias term are set to 0. The last layer has as weigths the parameters Alpha_out_layer 
# and its bias term b_out_layer

model_1_Inter = Classifier_NN_NObSP(model_1, torch.t(Alpha_NN_1), torch.t(Alpha_out_layer), b_out_layer,p) # Creating the Interpretable model
print(f'The bias per class are: {b_out_layer}')

# Evalauting the model   
model_1_Inter.eval() # Setting th emodel in evaluation mode
with torch.inference_mode():
    y_est_1_Inter, y_est_1_Inter_dec = model_1_Inter(X) # Computing th eoutput of the Interpretable model, the estimated final output and the decomposition

# Plotting the results for the estimated output
for i in range(p):
    
    fig, axs = plt.subplots(4,1)
    plt.tight_layout()
    axs[0].plot(t, y_est_1_Inter[:,i].cpu(), t, y_prob_1[:,i].cpu())  # Estimated output of the interpretable model vs the original model
    axs[1].scatter(y_prob_1[:,i].cpu(), y_est_1_Inter[:,i].cpu(), marker='x')  # Calibration plot between estimated output of the original model vs estimated output of the Interpretable model
    axs[2].plot(t, y[:,i], t, y_est_1_Inter[:,i].cpu()) # Estimated output of the interpretable model vs real output
    axs[3].scatter(y[:,i], y_est_1_Inter[:,i].cpu(), marker='x') # Calibration plot between estimated output of the Interpretable model vs real output
    #plt.show()

# Plotting the results for the estimated decomposition real and using the model
for i in range(p):
    
    fig, axs = plt.subplots(2,3)
    plt.tight_layout()
    axs[0,0].scatter(x0,y_est_1_Inter_dec[:,i*in_feat+0]-y_est_1_Inter_dec[:,i*in_feat+0].mean(),marker='o') # Estimated nonlinear contribution of x0 on the output
    axs[0,1].scatter(x1,y_est_1_Inter_dec[:,i*in_feat+1]-y_est_1_Inter_dec[:,i*in_feat+1].mean(),marker='o') # Estimated nonlinear contribution of x1 on the output
    axs[0,2].scatter(x2,y_est_1_Inter_dec[:,i*in_feat+2]-y_est_1_Inter_dec[:,i*in_feat+2].mean(),marker='o') # Estimated nonlinear contribution of x2 on the output
    axs[1,0].scatter(x3,y_est_1_Inter_dec[:,i*in_feat+3]-y_est_1_Inter_dec[:,i*in_feat+3].mean(),marker='o') # Estimated nonlinear contribution of x3 on the output
    axs[1,1].scatter(y[:,i].cpu(),y_est_1_Inter[:,i].cpu(), marker='x') # Calinbration plot real output vs estiamted output
    #plt.show()

# Estimation error in the projections

error_Model = torch.zeros((N,p))
error_Model_Approx = torch.zeros((N,p))
error_Approx = torch.zeros((N,p))

for i in range(p):
    error_Model[:,i] = y[:,i]-y_prob_1[:,i]
    error_Model_Approx[:,i] = y[:,i]-y_est_1_Inter[:,i]
    error_Approx [:,i]= y_prob_1[:,i]-y_est_1_Inter[:,i]

# Plotting the errors
for i in range(p):
    plt.tight_layout()
    fig, axs = plt.subplots(3,1)
    axs[0].plot(error_Model[:,i])
    axs[1].plot(error_Model_Approx[:,i])
    axs[2].plot(error_Approx[:,i])
    #plt.show()

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
    
# Final plots paper

# Defining colors    
C_blue = (21/255,76/255,121/255)
C_red = (135/255,12/255,15/255)
C_green = (21/255,121/255,76/255)
C_neutral = (0.2,0.2,0.2, 0.9)
C_grayish_back = (238/255, 238/255, 241/255, 0.05)

# Indexes for sorted values of input
x0_in = np.argsort(x0)
x1_in = np.argsort(x1)
x2_in = np.argsort(x2)
x3_in = np.argsort(x3)

fig, axs = plt.subplot_mosaic('AABBCC;AABBCC;DEFGHI;JKLMNO',figsize=(16,8)) # Creating the subplots mosaic
# Ploting the results for the setosa variety
axs['A'].plot(y_est_1_Inter[:,0], color=C_blue, linewidth=1, label='Estimated Output')
axs['A'].plot(y[:,0], color=C_neutral, linewidth=2, linestyle='--', label='Real Output')
axs['A'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['A'].legend()
axs['A'].set_xlim(left=0, right=150)
axs['A'].set_xlabel('Samples', fontsize=16, fontfamily='Baskerville')
axs['A'].set_ylabel('Probability', fontsize=16, fontfamily='Baskerville')
axs['A'].set_title('Setosa', fontsize=24, fontfamily='Baskerville', fontweight='black')

# Ploting the results for the versicolor variety
axs['B'].plot(y_est_1_Inter[:,1], color=C_red, linewidth=1, label='Estimated Output')
axs['B'].plot(y[:,1], color=C_neutral, linewidth=2, linestyle='--', label='Real Output')
axs['B'].grid(visible=True, alpha=0.3, linestyle=':')
axs['B'].set_facecolor(color=C_grayish_back)
axs['B'].legend(loc=2)
axs['B'].set_xlim(left=0, right=150)
axs['B'].set_xlabel('Samples', fontsize=16, fontfamily='Baskerville')
axs['B'].set_ylabel('Probability', fontsize=16, fontfamily='Baskerville')
axs['B'].set_title('Versicolor', fontsize=24, fontfamily='Baskerville', fontweight='black')

# Ploting the results for the virginica variety
axs['C'].plot(y_est_1_Inter[:,2], color=C_green, linewidth=1, label='Estimated Output')
axs['C'].plot(y[:,2], color=C_neutral, linewidth=2, linestyle='--', label='Real Output')
axs['C'].grid(visible=True, alpha=0.3, linestyle=':')
axs['C'].set_facecolor(color=C_grayish_back)
axs['C'].legend()
axs['C'].set_xlim(left=0, right=150)
axs['C'].set_xlabel('Samples', fontsize=16, fontfamily='Baskerville')
axs['C'].set_ylabel('Probability', fontsize=16, fontfamily='Baskerville')
axs['C'].set_title('Virginica', fontsize=24, fontfamily='Baskerville', fontweight='black')

#_________________________________________________________________________________________________
# Ploting contributions for sepal Length in Setosa
axs['D'].plot(x0[x0_in],y_est_1_Inter_dec[x0_in,0], color=C_blue, linewidth=1)
axs['D'].fill_between(x0[x0_in],y_est_1_Inter_dec[x0_in,0], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x0_in,0]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['D'].fill_between(x0[x0_in],y_est_1_Inter_dec[x0_in,0], color=C_blue, alpha=0.25,
                      where=(y_est_1_Inter_dec[x0_in,0]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['D'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['D'].set_xlabel('Sepal Length [cm]', fontsize=14, fontfamily='Baskerville')
axs['D'].set_ylabel('Contribution', fontsize=14, fontfamily='Baskerville')
axs['D'].set_xlim(left=x0.min(), right=x0.max())

# Ploting contributions for sepal Width in Setosa
axs['E'].plot(x1[x1_in],y_est_1_Inter_dec[x1_in,1], color=C_blue, linewidth=1)
axs['E'].fill_between(x1[x1_in],y_est_1_Inter_dec[x1_in,1], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x1_in,1]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['E'].fill_between(x1[x1_in],y_est_1_Inter_dec[x1_in,1], color=C_blue, alpha=0.25,
                      where=(y_est_1_Inter_dec[x1_in,1]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['E'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['E'].set_xlabel('Sepal Width [cm]', fontsize=14, fontfamily='Baskerville')
axs['E'].set_xlim(left=x1.min(), right=x1.max())

# Ploting contributions for Petal Length in Setosa
axs['J'].plot(x2[x2_in],y_est_1_Inter_dec[x2_in,2], color=C_blue, linewidth=1)
axs['J'].fill_between(x2[x2_in],y_est_1_Inter_dec[x2_in,2], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x2_in,2]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['J'].fill_between(x2[x2_in],y_est_1_Inter_dec[x2_in,2], color=C_blue, alpha=0.25,
                      where=(y_est_1_Inter_dec[x2_in,2]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['J'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['J'].set_xlabel('Petal Length [cm]', fontsize=14, fontfamily='Baskerville')
axs['J'].set_ylabel('Contribution', fontsize=14, fontfamily='Baskerville')
axs['J'].set_xlim(left=x2.min(), right=x2.max())

# Ploting contributions for sepal Width in Setosa
axs['K'].plot(x3[x3_in],y_est_1_Inter_dec[x3_in,3], color=C_blue, linewidth=1)
axs['K'].fill_between(x3[x3_in],y_est_1_Inter_dec[x3_in,3], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x3_in,3]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['K'].fill_between(x3[x3_in],y_est_1_Inter_dec[x3_in,3], color=C_blue, alpha=0.25,
                      where=(y_est_1_Inter_dec[x3_in,3]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['K'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['K'].set_xlabel('Petal width [cm]', fontsize=14, fontfamily='Baskerville')
axs['K'].set_xlim(left=x3.min(), right=x3.max())

#_________________________________________________________________________________________________
# Ploting contributions for sepal Length in Versicolor
axs['F'].plot(x0[x0_in],y_est_1_Inter_dec[x0_in,4], color=C_red, linewidth=1)
axs['F'].fill_between(x0[x0_in],y_est_1_Inter_dec[x0_in,4], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x0_in,4]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['F'].fill_between(x0[x0_in],y_est_1_Inter_dec[x0_in,4], color=C_red, alpha=0.25,
                      where=(y_est_1_Inter_dec[x0_in,4]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['F'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['F'].set_xlabel('Sepal Length [cm]', fontsize=14, fontfamily='Baskerville')
axs['F'].set_xlim(left=x0.min(), right=x0.max())

# Ploting contributions for sepal Width in Versicolor
axs['G'].plot(x1[x1_in],y_est_1_Inter_dec[x1_in,5], color=C_red, linewidth=1)
axs['G'].fill_between(x1[x1_in],y_est_1_Inter_dec[x1_in,5], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x1_in,5]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['G'].fill_between(x1[x1_in],y_est_1_Inter_dec[x1_in,5], color=C_red, alpha=0.25,
                      where=(y_est_1_Inter_dec[x1_in,5]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['G'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['G'].set_xlabel('Sepal Width [cm]', fontsize=14, fontfamily='Baskerville')
axs['G'].set_xlim(left=x1.min(), right=x1.max())

# Ploting contributions for Petal Length in Versicolor
axs['L'].plot(x2[x2_in],y_est_1_Inter_dec[x2_in,6], color=C_red, linewidth=1)
axs['L'].fill_between(x2[x2_in],y_est_1_Inter_dec[x2_in,6], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x0_in,6]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['L'].fill_between(x2[x2_in],y_est_1_Inter_dec[x2_in,6], color=C_red, alpha=0.25,
                      where=(y_est_1_Inter_dec[x2_in,6]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['L'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['L'].set_xlabel('Petal Length [cm]', fontsize=14, fontfamily='Baskerville')
axs['L'].set_xlim(left=x2.min(), right=x2.max())

# Ploting contributions for sepal Width in Versicolor
axs['M'].plot(x3[x3_in],y_est_1_Inter_dec[x3_in,7], color=C_red, linewidth=1)
axs['M'].fill_between(x3[x3_in],y_est_1_Inter_dec[x3_in,7], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x3_in,7]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['M'].fill_between(x3[x3_in],y_est_1_Inter_dec[x3_in,7], color=C_red, alpha=0.25,
                      where=(y_est_1_Inter_dec[x3_in,7]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['M'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['M'].set_xlabel('Petal width [cm]', fontsize=14, fontfamily='Baskerville')
axs['M'].set_xlim(left=x3.min(), right=x3.max())

#_________________________________________________________________________________________________
# Ploting contributions for sepal Length in Virginica
axs['H'].plot(x0[x0_in],y_est_1_Inter_dec[x0_in,8], color=C_green, linewidth=1)
axs['H'].fill_between(x0[x0_in],y_est_1_Inter_dec[x0_in,8], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x0_in,8]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['H'].fill_between(x0[x0_in],y_est_1_Inter_dec[x0_in,8], color=C_green, alpha=0.25,
                      where=(y_est_1_Inter_dec[x0_in,8]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['H'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['H'].set_xlabel('Sepal Length [cm]', fontsize=14, fontfamily='Baskerville')
axs['H'].set_xlim(left=x0.min(), right=x0.max())

# Ploting contributions for sepal Width in Virginica
axs['I'].plot(x1[x1_in],y_est_1_Inter_dec[x1_in,9], color=C_green, linewidth=1)
axs['I'].fill_between(x1[x1_in],y_est_1_Inter_dec[x1_in,9], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x1_in,9]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['I'].fill_between(x1[x1_in],y_est_1_Inter_dec[x1_in,9], color=C_green, alpha=0.25,
                      where=(y_est_1_Inter_dec[x1_in,9]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['I'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['I'].set_xlabel('Sepal Width [cm]', fontsize=14, fontfamily='Baskerville')
axs['I'].set_xlim(left=x1.min(), right=x1.max())

# Ploting contributions for Petal Length in Virginica
axs['N'].plot(x2[x2_in],y_est_1_Inter_dec[x2_in,10], color=C_green, linewidth=1)
axs['N'].fill_between(x2[x2_in],y_est_1_Inter_dec[x2_in,10], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x2_in,10]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['N'].fill_between(x2[x2_in],y_est_1_Inter_dec[x2_in,10], color=C_green, alpha=0.25,
                      where=(y_est_1_Inter_dec[x2_in,10]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['N'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['N'].set_xlabel('Petal Length [cm]', fontsize=14, fontfamily='Baskerville')
axs['N'].set_xlim(left=x2.min(), right=x2.max())

# Ploting contributions for sepal Width in Virginica
axs['O'].plot(x3[x3_in],y_est_1_Inter_dec[x3_in,11], color=C_green, linewidth=1)
axs['O'].fill_between(x3[x3_in],y_est_1_Inter_dec[x3_in,11], color=(1,1,1),
                      where=(y_est_1_Inter_dec[x3_in,11]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['O'].fill_between(x3[x3_in],y_est_1_Inter_dec[x3_in,11], color=C_green, alpha=0.25,
                      where=(y_est_1_Inter_dec[x3_in,11]>0), interpolate=True, edgecolor=(1,1,1,0))
axs['O'].grid(visible=True, alpha=0.3, linestyle=':')
axs['A'].set_facecolor(color=C_grayish_back)
axs['O'].set_xlabel('Petal width [cm]', fontsize=14, fontfamily='Baskerville')
axs['O'].set_xlim(left=x3.min(), right=x3.max())

plt.tight_layout()
#plt.show()

# Generating test data

# Creating vectors for the interpolation
n_e = 100
x0_b = np.linspace(4,8,n_e)
x1_b = np.linspace(1.9,4.5,n_e)
x2_b = np.linspace(0.9,7.1,n_e)
x3_b = np.linspace(0,2.6,n_e)
alpha = 0.008
# Creating the meshgrid

X0m, X1m = np.meshgrid(x0_b, x1_b)
X2m, X3m = np.meshgrid(x2_b, x3_b)
X3m_t, X2m_t = np.meshgrid(x3_b, x2_b)

X0m_resh = X0m.reshape((n_e*n_e,))
X1m_resh = X1m.reshape((n_e*n_e,))
X2m_resh = X2m.reshape((n_e*n_e,))
X3m_resh = X3m.reshape((n_e*n_e,))
X2m_t_resh = X2m_t.reshape((n_e*n_e,))
X3m_t_resh = X3m_t.reshape((n_e*n_e,))

# Obtaining the original coordinates sorted
x0_sort = x0[x0_in]
x1_sort = x1[x1_in]
x2_sort = x2[x2_in]
x3_sort = x3[x3_in]

# Thresholds per class
# Th_0 = -b_out_layer[0,0].cpu()/in_feat
# Th_1 = -b_out_layer[0,1].cpu()/in_feat
# Th_2 = -b_out_layer[0,2].cpu()/in_feat
Th_0 = 0
Th_1 = 0
Th_2 = 0

# Estimation for the Setosa Class
des_x0_0 = np.zeros((len(x0_sort),)) # Initializing vector for class 0 to set values when it is equal to 1
des_x0_0[y_est_1_Inter_dec[x0_in,0]>Th_0] = 1 #  Obtaining vector of desistion for class 0 based on its decomposition
inter_des_x00 = np.interp(x0_b,x0_sort,des_x0_0) # Obtaining interpolated output
inter_des_x00[inter_des_x00>=0.5] = 1 # Preprocessing the utput to have values 0 and 1
inter_des_x00[inter_des_x00<0.5] = 0 # Preprocessing the utput to have values 0 and 1
inter_des_x00 = inter_des_x00[:,np.newaxis]

des_x1_0 = np.zeros((len(x1_sort),))
des_x1_0[y_est_1_Inter_dec[x1_in,1]>Th_0] = 1
inter_des_x10 = np.interp(x1_b,x1_sort,des_x1_0)
inter_des_x10[inter_des_x10>=0.5] = 1
inter_des_x10[inter_des_x10<0.5] = 0
inter_des_x10 = inter_des_x10[:,np.newaxis]

des_x2_0 = np.zeros((len(x2_sort),))
des_x2_0[y_est_1_Inter_dec[x2_in,2]>Th_0] = 1
inter_des_x20 = np.interp(x2_b,x2_sort,des_x2_0)
inter_des_x20[inter_des_x20>=0.5] = 1
inter_des_x20[inter_des_x20<0.5] = 0
inter_des_x20 = inter_des_x20[:,np.newaxis]

des_x3_0 = np.zeros((len(x3_sort),))
des_x3_0[y_est_1_Inter_dec[x3_in,3]>Th_0] = 1
inter_des_x30 = np.interp(x3_b,x3_sort,des_x3_0)
inter_des_x30[inter_des_x30>=0.5] = 1
inter_des_x30[inter_des_x30<0.5] = 0
inter_des_x30 = inter_des_x30[:,np.newaxis]

# Estimation for the Versicolor Class
des_x0_1 = np.zeros((len(x0_sort),)) # Initializing vector for class 0 to set values when it is equal to 1
des_x0_1[y_est_1_Inter_dec[x0_in,4]>Th_1] = 1 #  Obtaining vector of desistion for class 0 based on its decomposition
inter_des_x01 = np.interp(x0_b,x0_sort,des_x0_1) # Obtaining interpolated output
inter_des_x01[inter_des_x01>=0.5] = 1 # Preprocessing the utput to have values 0 and 1
inter_des_x01[inter_des_x01<0.5] = 0 # Preprocessing the utput to have values 0 and 1
inter_des_x01 = inter_des_x01[:,np.newaxis]

des_x1_1 = np.zeros((len(x1_sort),))
des_x1_1[y_est_1_Inter_dec[x1_in,5]>Th_1] = 1
inter_des_x11 = np.interp(x1_b,x1_sort,des_x1_1)
inter_des_x11[inter_des_x11>=0.5] = 1
inter_des_x11[inter_des_x11<0.5] = 0
inter_des_x11 = inter_des_x11[:,np.newaxis]

des_x2_1 = np.zeros((len(x2_sort),))
des_x2_1[y_est_1_Inter_dec[x2_in,6]>Th_1] = 1
inter_des_x21 = np.interp(x2_b,x2_sort,des_x2_1)
inter_des_x21[inter_des_x21>=0.5] = 1
inter_des_x21[inter_des_x21<0.5] = 0
inter_des_x21 = inter_des_x21[:,np.newaxis]

des_x3_1 = np.zeros((len(x3_sort),))
des_x3_1[y_est_1_Inter_dec[x3_in,7]>Th_1] = 1
inter_des_x31 = np.interp(x3_b,x3_sort,des_x3_1)
inter_des_x31[inter_des_x31>=0.5] = 1
inter_des_x31[inter_des_x31<0.5] = 0
inter_des_x31 = inter_des_x31[:,np.newaxis]

# Estimation for the Virginica Class
des_x0_3 = np.zeros((len(x0_sort),)) # Initializing vector for class 0 to set values when it is equal to 1
des_x0_3[y_est_1_Inter_dec[x0_in,8]>Th_2] = 1 #  Obtaining vector of desistion for class 0 based on its decomposition
inter_des_x03 = np.interp(x0_b,x0_sort,des_x0_3) # Obtaining interpolated output
inter_des_x03[inter_des_x03>=0.5] = 1 # Preprocessing the utput to have values 0 and 1
inter_des_x03[inter_des_x03<0.5] = 0 # Preprocessing the utput to have values 0 and 1
inter_des_x03 = inter_des_x03[:,np.newaxis]

des_x1_3 = np.zeros((len(x1_sort),))
des_x1_3[y_est_1_Inter_dec[x1_in,9]>Th_2] = 1
inter_des_x13 = np.interp(x1_b,x1_sort,des_x1_3)
inter_des_x13[inter_des_x13>=0.5] = 1
inter_des_x13[inter_des_x13<0.5] = 0
inter_des_x13 = inter_des_x13[:,np.newaxis]

des_x2_3 = np.zeros((len(x2_sort),))
des_x2_3[y_est_1_Inter_dec[x2_in,10]>Th_2] = 1
inter_des_x23 = np.interp(x2_b,x2_sort,des_x2_3)
inter_des_x23[inter_des_x23>=0.5] = 1
inter_des_x23[inter_des_x23<0.5] = 0
inter_des_x23 = inter_des_x23[:,np.newaxis]

des_x3_3 = np.zeros((len(x3_sort),))
des_x3_3[y_est_1_Inter_dec[x3_in,11]>Th_2] = 1
inter_des_x33 = np.interp(x3_b,x3_sort,des_x3_3)
inter_des_x33[inter_des_x33>=0.5] = 1
inter_des_x33[inter_des_x33<0.5] = 0
inter_des_x33 = inter_des_x33[:,np.newaxis]

fig, axs = plt.subplots(4,4,figsize=(12,12)) # Creating the subplots mosaic

# Ploting distributions

bandwidth = 0.1 # Parameter for kernel estimation

# For Sepal length
dx0_1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx0_1.fit(x0[0:50, None])
logprob_1 = dx0_1.score_samples(x0_b[:, None])

dx0_2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx0_2.fit(x0[50:100, None])
logprob_2 = dx0_2.score_samples(x0_b[:, None])

dx0_3 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx0_3.fit(x0[100:150, None])
logprob_3 = dx0_3.score_samples(x0_b[:, None])

axs[0,0].fill_between(x0_b, np.exp(logprob_1), color=C_blue,  label= 'Setosa', alpha=0.2)
axs[0,0].fill_between(x0_b, np.exp(logprob_2), color=C_red,  label= 'Versicolor', alpha=0.2)
axs[0,0].fill_between(x0_b, np.exp(logprob_3), color=C_green,  label= 'Virginica', alpha=0.2)
axs[0,0].grid(visible=True, alpha=0.3, linestyle=':')
axs[0,0].set_facecolor(color=C_grayish_back)
axs[0,0].set_xlim(left=x0_b.min(), right=x0_b.max())
axs[0,0].legend()

# For Sepal width
dx1_1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx1_1.fit(x1[0:50, None])
logprob_1 = dx1_1.score_samples(x1_b[:, None])

dx1_2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx1_2.fit(x1[50:100, None])
logprob_2 = dx1_2.score_samples(x1_b[:, None])

dx1_3 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx1_3.fit(x1[100:150, None])
logprob_3 = dx1_3.score_samples(x1_b[:, None])

axs[1,1].fill_between(x1_b, np.exp(logprob_1), color=C_blue,  label= 'Setosa', alpha=0.2)
axs[1,1].fill_between(x1_b, np.exp(logprob_2), color=C_red,  label= 'Versicolor', alpha=0.2)
axs[1,1].fill_between(x1_b, np.exp(logprob_3), color=C_green,  label= 'Virginica', alpha=0.2)
axs[1,1].grid(visible=True, alpha=0.3, linestyle=':')
axs[1,1].set_facecolor(color=C_grayish_back)
axs[1,1].set_xlim(left=x1_b.min(), right=x1_b.max())

# For Petal length
dx2_1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx2_1.fit(x2[0:50, None])
logprob_1 = dx2_1.score_samples(x2_b[:, None])

dx2_2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx2_2.fit(x2[50:100, None])
logprob_2 = dx2_2.score_samples(x2_b[:, None])

dx2_3 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx2_3.fit(x2[100:150, None])
logprob_3 = dx2_3.score_samples(x2_b[:, None])

axs[2,2].fill_between(x2_b, np.exp(logprob_1), color=C_blue,  label= 'Setosa', alpha=0.2)
axs[2,2].fill_between(x2_b, np.exp(logprob_2), color=C_red,  label= 'Versicolor', alpha=0.2)
axs[2,2].fill_between(x2_b, np.exp(logprob_3), color=C_green,  label= 'Virginica', alpha=0.2)
axs[2,2].grid(visible=True, alpha=0.3, linestyle=':')
axs[2,2].set_facecolor(color=C_grayish_back)
axs[2,2].set_xlim(left=x2_b.min(), right=x2_b.max())

# For Petal width
dx3_1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx3_1.fit(x3[0:50, None])
logprob_1 = dx3_1.score_samples(x3_b[:, None])

dx3_2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx3_2.fit(x3[50:100, None])
logprob_2 = dx3_2.score_samples(x3_b[:, None])

dx3_3 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx3_3.fit(x3[100:150, None])
logprob_3 = dx3_3.score_samples(x3_b[:, None])

axs[3,3].fill_between(x3_b, np.exp(logprob_1), color=C_blue,  label= 'Setosa', alpha=0.2)
axs[3,3].fill_between(x3_b, np.exp(logprob_2), color=C_red,  label= 'Versicolor', alpha=0.2)
axs[3,3].fill_between(x3_b, np.exp(logprob_3), color=C_green,  label= 'Virginica', alpha=0.2)
axs[3,3].grid(visible=True, alpha=0.3, linestyle=':')
axs[3,3].set_facecolor(color=C_grayish_back)
axs[3,3].set_xlim(left=x3_b.min(), right=x3_b.max())
axs[3,3].set_xlabel('Petal width [cm]', fontsize=14, fontfamily='Baskerville')

# Relation between Sepal size and width
D_01_1 = inter_des_x10@np.transpose(inter_des_x00)
D_01_1 = D_01_1.reshape((n_e*n_e,))
D_01_2 = inter_des_x11@np.transpose(inter_des_x01)
D_01_2 = D_01_2.reshape((n_e*n_e,))
D_01_3 = inter_des_x13@np.transpose(inter_des_x03)
D_01_3 = D_01_3.reshape((n_e*n_e,))

axs[0,1].scatter(x1[0:50],x0[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[0,1].scatter(x1[50:100],x0[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[0,1].scatter(x1[100:150],x0[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[0,1].grid(visible=True, alpha=0.3, linestyle=':')
axs[0,1].set_facecolor(color=C_grayish_back)
axs[0,1].set_ylim(bottom=4, top=8.5)

axs[0,1].scatter(X1m_resh[D_01_1==1], X0m_resh[D_01_1==1], color=C_blue, alpha=alpha)
axs[0,1].scatter(X1m_resh[D_01_2==1], X0m_resh[D_01_2==1], color=C_red, alpha=alpha)
axs[0,1].scatter(X1m_resh[D_01_3==1], X0m_resh[D_01_3==1], color=C_green, alpha=alpha)
axs[0,1].set_ylim(bottom=x0_b.min(), top=x0_b.max())
axs[0,1].set_xlim(left=x1_b.min(), right=x1_b.max())

axs[1,0].scatter(x0[0:50],x1[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[1,0].scatter(x0[50:100],x1[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[1,0].scatter(x0[100:150],x1[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[1,0].grid(visible=True, alpha=0.3, linestyle=':')
axs[1,0].set_facecolor(color=C_grayish_back)
axs[1,0].set_ylabel('Sepal width [cm]', fontsize=14, fontfamily='Baskerville')

axs[1,0].scatter(X0m_resh[D_01_1==1], X1m_resh[D_01_1==1], color=C_blue, alpha=alpha)
axs[1,0].scatter(X0m_resh[D_01_2==1], X1m_resh[D_01_2==1], color=C_red, alpha=alpha)
axs[1,0].scatter(X0m_resh[D_01_3==1], X1m_resh[D_01_3==1], color=C_green, alpha=alpha)
axs[1,0].set_ylim(bottom=x1_b.min(), top=x1_b.max())
axs[1,0].set_xlim(left=x0_b.min(), right=x0_b.max())

# Relation between Sepal size and Petal Size
D_02_1 = inter_des_x20@np.transpose(inter_des_x00)
D_02_1 = D_02_1.reshape((n_e*n_e,))
D_02_2 = inter_des_x21@np.transpose(inter_des_x01)
D_02_2 = D_02_2.reshape((n_e*n_e,))
D_02_3 = inter_des_x23@np.transpose(inter_des_x03)
D_02_3 = D_02_3.reshape((n_e*n_e,))

axs[0,2].scatter(x2[0:50],x0[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[0,2].scatter(x2[50:100],x0[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[0,2].scatter(x2[100:150],x0[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[0,2].grid(visible=True, alpha=0.3, linestyle=':')
axs[0,2].set_facecolor(color=C_grayish_back)

axs[0,2].scatter(X2m_t_resh[D_02_1==1], X0m_resh[D_02_1==1], color=C_blue, alpha=alpha)
axs[0,2].scatter(X2m_t_resh[D_02_2==1], X0m_resh[D_02_2==1], color=C_red, alpha=alpha)
axs[0,2].scatter(X2m_t_resh[D_02_3==1], X0m_resh[D_02_3==1], color=C_green, alpha=alpha)
axs[0,2].set_ylim(bottom=x0_b.min(), top=x0_b.max())
axs[0,2].set_xlim(left=x2_b.min(), right=x2_b.max())

axs[2,0].scatter(x0[0:50],x2[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[2,0].scatter(x0[50:100],x2[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[2,0].scatter(x0[100:150],x2[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[2,0].grid(visible=True, alpha=0.3, linestyle=':')
axs[2,0].set_facecolor(color=C_grayish_back)
axs[2,0].set_ylabel('Petal Length [cm]', fontsize=14, fontfamily='Baskerville')

axs[2,0].scatter(X0m_resh[D_02_1==1], X2m_t_resh[D_02_1==1], color=C_blue, alpha=alpha)
axs[2,0].scatter(X0m_resh[D_02_2==1], X2m_t_resh[D_02_2==1], color=C_red, alpha=alpha)
axs[2,0].scatter(X0m_resh[D_02_3==1], X2m_t_resh[D_02_3==1], color=C_green, alpha=alpha)
axs[2,0].set_ylim(bottom=x2_b.min(), top=x2_b.max())
axs[2,0].set_xlim(left=x0_b.min(), right=x0_b.max())

# Relation between Sepal size and Petal width
D_03_1 = inter_des_x30@np.transpose(inter_des_x00)
D_03_1 = D_03_1.reshape((n_e*n_e,))
D_03_2 = inter_des_x31@np.transpose(inter_des_x01)
D_03_2 = D_03_2.reshape((n_e*n_e,))
D_03_3 = inter_des_x33@np.transpose(inter_des_x03)
D_03_3 = D_03_3.reshape((n_e*n_e,))

axs[0,3].scatter(x3[0:50],x0[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[0,3].scatter(x3[50:100],x0[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[0,3].scatter(x3[100:150],x0[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[0,3].grid(visible=True, alpha=0.3, linestyle=':')
axs[0,3].set_facecolor(color=C_grayish_back)

axs[0,3].scatter(X3m_resh[D_03_1==1], X0m_resh[D_03_1==1], color=C_blue, alpha=alpha)
axs[0,3].scatter(X3m_resh[D_03_2==1], X0m_resh[D_03_2==1], color=C_red, alpha=alpha)
axs[0,3].scatter(X3m_resh[D_03_3==1], X0m_resh[D_03_3==1], color=C_green, alpha=alpha)
axs[0,3].set_ylim(bottom=x0_b.min(), top=x0_b.max())
axs[0,3].set_xlim(left=x3_b.min(), right=x3_b.max())

axs[3,0].scatter(x0[0:50],x3[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[3,0].scatter(x0[50:100],x3[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[3,0].scatter(x0[100:150],x3[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[3,0].grid(visible=True, alpha=0.3, linestyle=':')
axs[3,0].set_facecolor(color=C_grayish_back)
axs[3,0].set_ylabel('Petal width [cm]', fontsize=14, fontfamily='Baskerville')
axs[3,0].set_xlabel('Sepal length [cm]', fontsize=14, fontfamily='Baskerville')

axs[3,0].scatter(X0m_resh[D_03_1==1], X3m_resh[D_03_1==1], color=C_blue, alpha=alpha)
axs[3,0].scatter(X0m_resh[D_03_2==1], X3m_resh[D_03_2==1], color=C_red, alpha=alpha)
axs[3,0].scatter(X0m_resh[D_03_3==1], X3m_resh[D_03_3==1], color=C_green, alpha=alpha)
axs[3,0].set_ylim(bottom=x3_b.min(), top=x3_b.max())
axs[3,0].set_xlim(left=x0_b.min(), right=x0_b.max())

# Relation between Sepal Width and Petal length
D_12_1 = inter_des_x10@np.transpose(inter_des_x20)
D_12_1 = D_12_1.reshape((n_e*n_e,))
D_12_2 = inter_des_x11@np.transpose(inter_des_x21)
D_12_2 = D_12_2.reshape((n_e*n_e,))
D_12_3 = inter_des_x13@np.transpose(inter_des_x23)
D_12_3 = D_12_3.reshape((n_e*n_e,))

axs[1,2].scatter(x2[0:50],x1[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[1,2].scatter(x2[50:100],x1[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[1,2].scatter(x2[100:150],x1[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[1,2].grid(visible=True, alpha=0.3, linestyle=':')
axs[1,2].set_facecolor(color=C_grayish_back)

axs[1,2].scatter(X2m_resh[D_12_1==1], X1m_resh[D_12_1==1], color=C_blue, alpha=alpha)
axs[1,2].scatter(X2m_resh[D_12_2==1], X1m_resh[D_12_2==1], color=C_red, alpha=alpha)
axs[1,2].scatter(X2m_resh[D_12_3==1], X1m_resh[D_12_3==1], color=C_green, alpha=alpha)
axs[1,2].set_ylim(bottom=x1_b.min(), top=x1_b.max())
axs[1,2].set_xlim(left=x2_b.min(), right=x2_b.max())

axs[2,1].scatter(x1[0:50],x2[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[2,1].scatter(x1[50:100],x2[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[2,1].scatter(x1[100:150],x2[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[2,1].grid(visible=True, alpha=0.3, linestyle=':')
axs[2,1].set_facecolor(color=C_grayish_back)

axs[2,1].scatter(X1m_resh[D_12_1==1], X2m_resh[D_12_1==1], color=C_blue, alpha=alpha)
axs[2,1].scatter(X1m_resh[D_12_2==1], X2m_resh[D_12_2==1], color=C_red, alpha=alpha)
axs[2,1].scatter(X1m_resh[D_12_3==1], X2m_resh[D_12_3==1], color=C_green, alpha=alpha)
axs[2,1].set_ylim(bottom=x2_b.min(), top=x2_b.max())
axs[2,1].set_xlim(left=x1_b.min(), right=x1_b.max())

# Relation between Sepal Width and Petal width
D_13_1 = inter_des_x10@np.transpose(inter_des_x30)
D_13_1 = D_13_1.reshape((n_e*n_e,))
D_13_2 = inter_des_x11@np.transpose(inter_des_x31)
D_13_2 = D_13_2.reshape((n_e*n_e,))
D_13_3 = inter_des_x13@np.transpose(inter_des_x33)
D_13_3 = D_13_3.reshape((n_e*n_e,))

axs[1,3].scatter(x3[0:50],x1[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[1,3].scatter(x3[50:100],x1[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[1,3].scatter(x3[100:150],x1[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[1,3].grid(visible=True, alpha=0.3, linestyle=':')
axs[1,3].set_facecolor(color=C_grayish_back)

axs[1,3].scatter(X3m_t_resh[D_13_1==1], X1m_resh[D_13_1==1], color=C_blue, alpha=alpha)
axs[1,3].scatter(X3m_t_resh[D_13_2==1], X1m_resh[D_13_2==1], color=C_red, alpha=alpha)
axs[1,3].scatter(X3m_t_resh[D_13_3==1], X1m_resh[D_13_3==1], color=C_green, alpha=alpha)
axs[1,3].set_ylim(bottom=x1_b.min(), top=x1_b.max())
axs[1,3].set_xlim(left=x3_b.min(), right=x3_b.max())

axs[3,1].scatter(x1[0:50],x3[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[3,1].scatter(x1[50:100],x3[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[3,1].scatter(x1[100:150],x3[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[3,1].grid(visible=True, alpha=0.3, linestyle=':')
axs[3,1].set_facecolor(color=C_grayish_back)
axs[3,1].set_xlabel('Sepal width [cm]', fontsize=14, fontfamily='Baskerville')

axs[3,1].scatter(X1m_resh[D_13_1==1], X3m_t_resh[D_13_1==1], color=C_blue, alpha=alpha)
axs[3,1].scatter(X1m_resh[D_13_2==1], X3m_t_resh[D_13_2==1], color=C_red, alpha=alpha)
axs[3,1].scatter(X1m_resh[D_13_3==1], X3m_t_resh[D_13_3==1], color=C_green, alpha=alpha)
axs[3,1].set_ylim(bottom=x3_b.min(), top=x3_b.max())
axs[3,1].set_xlim(left=x1_b.min(), right=x1_b.max())

# Relation between Petal length and Petal width
D_23_1 = inter_des_x20@np.transpose(inter_des_x30)
D_23_1 = D_23_1.reshape((n_e*n_e,))
D_23_2 = inter_des_x21@np.transpose(inter_des_x31)
D_23_2 = D_23_2.reshape((n_e*n_e,))
D_23_3 = inter_des_x23@np.transpose(inter_des_x33)
D_23_3 = D_23_3.reshape((n_e*n_e,))

axs[2,3].scatter(x3[0:50],x2[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[2,3].scatter(x3[50:100],x2[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[2,3].scatter(x3[100:150],x2[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[2,3].grid(visible=True, alpha=0.3, linestyle=':')
axs[2,3].set_facecolor(color=C_grayish_back)

axs[2,3].scatter(X3m_resh[D_23_1==1], X2m_resh[D_23_1==1], color=C_blue, alpha=alpha)
axs[2,3].scatter(X3m_resh[D_23_2==1], X2m_resh[D_23_2==1], color=C_red, alpha=alpha)
axs[2,3].scatter(X3m_resh[D_23_3==1], X2m_resh[D_23_3==1], color=C_green, alpha=alpha)
axs[2,3].set_ylim(bottom=x2_b.min(), top=x2_b.max())
axs[2,3].set_xlim(left=x3_b.min(), right=x3_b.max())

axs[3,2].scatter(x2[0:50],x3[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[3,2].scatter(x2[50:100],x3[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[3,2].scatter(x2[100:150],x3[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[3,2].grid(visible=True, alpha=0.3, linestyle=':')
axs[3,2].set_facecolor(color=C_grayish_back)
axs[3,2].set_xlabel('Petal length [cm]', fontsize=14, fontfamily='Baskerville')

axs[3,2].scatter(X2m_resh[D_23_1==1], X3m_resh[D_23_1==1], color=C_blue, alpha=alpha)
axs[3,2].scatter(X2m_resh[D_23_2==1], X3m_resh[D_23_2==1], color=C_red, alpha=alpha)
axs[3,2].scatter(X2m_resh[D_23_3==1], X3m_resh[D_23_3==1], color=C_green, alpha=alpha)
axs[3,2].set_ylim(bottom=x3_b.min(), top=x3_b.max())
axs[3,2].set_xlim(left=x2_b.min(), right=x2_b.max())

plt.tight_layout()
#plt.show()










# Generating test data

x0 = y_est_1_Inter_dec[:,0]
x1 = y_est_1_Inter_dec[:,1]
x2 = y_est_1_Inter_dec[:,2]
x3 = y_est_1_Inter_dec[:,3]

# Creating vectors for the interpolation
n_e = 100
x0_b = np.linspace(x0.min()-0.1*x0.min(),x0.max()+0.1*x0.max(),n_e)
x1_b = np.linspace(x1.min()-0.1*x1.min(),x1.max()+0.1*x1.max(),n_e)
x2_b = np.linspace(x2.min()-0.1*x2.min(),x2.max()+0.1*x2.max(),n_e)
x3_b = np.linspace(x3.min()-0.1*x3.min(),x3.max()+0.1*x3.max(),n_e)
alpha = 0.008
# Creating the meshgrid

X0m, X1m = np.meshgrid(x0_b, x1_b)
X2m, X3m = np.meshgrid(x2_b, x3_b)
X3m_t, X2m_t = np.meshgrid(x3_b, x2_b)

X0m_resh = X0m.reshape((n_e*n_e,))
X1m_resh = X1m.reshape((n_e*n_e,))
X2m_resh = X2m.reshape((n_e*n_e,))
X3m_resh = X3m.reshape((n_e*n_e,))
X2m_t_resh = X2m_t.reshape((n_e*n_e,))
X3m_t_resh = X3m_t.reshape((n_e*n_e,))



# Obtaining the original coordinates sorted
x0_sort = x0[x0_in]
x1_sort = x1[x1_in]
x2_sort = x2[x2_in]
x3_sort = x3[x3_in]

# Thresholds per class
Th_0 = -b_out_layer[0,0].cpu()/in_feat
Th_1 = -b_out_layer[0,1].cpu()/in_feat
Th_2 = -b_out_layer[0,2].cpu()/in_feat
print(Th_0)
print(Th_1)
print(Th_2)
print(Alpha_out_layer)

# Estimation for the Setosa Class
des_x0_0 = np.zeros((len(x0_sort),)) # Initializing vector for class 0 to set values when it is equal to 1
des_x0_0[y_est_1_Inter_dec[x0_in,0]>Th_0] = 1 #  Obtaining vector of desistion for class 0 based on its decomposition
inter_des_x00 = np.interp(x0_b,x0_sort,des_x0_0) # Obtaining interpolated output
inter_des_x00[inter_des_x00>=0.5] = 1 # Preprocessing the utput to have values 0 and 1
inter_des_x00[inter_des_x00<0.5] = 0 # Preprocessing the utput to have values 0 and 1
inter_des_x00 = inter_des_x00[:,np.newaxis]

des_x1_0 = np.zeros((len(x1_sort),))
des_x1_0[y_est_1_Inter_dec[x1_in,1]>Th_0] = 1
inter_des_x10 = np.interp(x1_b,x1_sort,des_x1_0)
inter_des_x10[inter_des_x10>=0.5] = 1
inter_des_x10[inter_des_x10<0.5] = 0
inter_des_x10 = inter_des_x10[:,np.newaxis]

des_x2_0 = np.zeros((len(x2_sort),))
des_x2_0[y_est_1_Inter_dec[x2_in,2]>Th_0] = 1
inter_des_x20 = np.interp(x2_b,x2_sort,des_x2_0)
inter_des_x20[inter_des_x20>=0.5] = 1
inter_des_x20[inter_des_x20<0.5] = 0
inter_des_x20 = inter_des_x20[:,np.newaxis]

des_x3_0 = np.zeros((len(x3_sort),))
des_x3_0[y_est_1_Inter_dec[x3_in,3]>Th_0] = 1
inter_des_x30 = np.interp(x3_b,x3_sort,des_x3_0)
inter_des_x30[inter_des_x30>=0.5] = 1
inter_des_x30[inter_des_x30<0.5] = 0
inter_des_x30 = inter_des_x30[:,np.newaxis]

# Estimation for the Versicolor Class
des_x0_1 = np.zeros((len(x0_sort),)) # Initializing vector for class 0 to set values when it is equal to 1
des_x0_1[y_est_1_Inter_dec[x0_in,4]>Th_1] = 1 #  Obtaining vector of desistion for class 0 based on its decomposition
inter_des_x01 = np.interp(x0_b,x0_sort,des_x0_1) # Obtaining interpolated output
inter_des_x01[inter_des_x01>=0.5] = 1 # Preprocessing the utput to have values 0 and 1
inter_des_x01[inter_des_x01<0.5] = 0 # Preprocessing the utput to have values 0 and 1
inter_des_x01 = inter_des_x01[:,np.newaxis]

des_x1_1 = np.zeros((len(x1_sort),))
des_x1_1[y_est_1_Inter_dec[x1_in,5]>Th_1] = 1
inter_des_x11 = np.interp(x1_b,x1_sort,des_x1_1)
inter_des_x11[inter_des_x11>=0.5] = 1
inter_des_x11[inter_des_x11<0.5] = 0
inter_des_x11 = inter_des_x11[:,np.newaxis]

des_x2_1 = np.zeros((len(x2_sort),))
des_x2_1[y_est_1_Inter_dec[x2_in,6]>Th_1] = 1
inter_des_x21 = np.interp(x2_b,x2_sort,des_x2_1)
inter_des_x21[inter_des_x21>=0.5] = 1
inter_des_x21[inter_des_x21<0.5] = 0
inter_des_x21 = inter_des_x21[:,np.newaxis]

des_x3_1 = np.zeros((len(x3_sort),))
des_x3_1[y_est_1_Inter_dec[x3_in,7]>Th_1] = 1
inter_des_x31 = np.interp(x3_b,x3_sort,des_x3_1)
inter_des_x31[inter_des_x31>=0.5] = 1
inter_des_x31[inter_des_x31<0.5] = 0
inter_des_x31 = inter_des_x31[:,np.newaxis]

# Estimation for the Virginica Class
des_x0_3 = np.zeros((len(x0_sort),)) # Initializing vector for class 0 to set values when it is equal to 1
des_x0_3[y_est_1_Inter_dec[x0_in,8]>Th_2] = 1 #  Obtaining vector of desistion for class 0 based on its decomposition
inter_des_x03 = np.interp(x0_b,x0_sort,des_x0_3) # Obtaining interpolated output
inter_des_x03[inter_des_x03>=0.5] = 1 # Preprocessing the utput to have values 0 and 1
inter_des_x03[inter_des_x03<0.5] = 0 # Preprocessing the utput to have values 0 and 1
inter_des_x03 = inter_des_x03[:,np.newaxis]

des_x1_3 = np.zeros((len(x1_sort),))
des_x1_3[y_est_1_Inter_dec[x1_in,9]>Th_2] = 1
inter_des_x13 = np.interp(x1_b,x1_sort,des_x1_3)
inter_des_x13[inter_des_x13>=0.5] = 1
inter_des_x13[inter_des_x13<0.5] = 0
inter_des_x13 = inter_des_x13[:,np.newaxis]

des_x2_3 = np.zeros((len(x2_sort),))
des_x2_3[y_est_1_Inter_dec[x2_in,10]>Th_2] = 1
inter_des_x23 = np.interp(x2_b,x2_sort,des_x2_3)
inter_des_x23[inter_des_x23>=0.5] = 1
inter_des_x23[inter_des_x23<0.5] = 0
inter_des_x23 = inter_des_x23[:,np.newaxis]

des_x3_3 = np.zeros((len(x3_sort),))
des_x3_3[y_est_1_Inter_dec[x3_in,11]>Th_2] = 1
inter_des_x33 = np.interp(x3_b,x3_sort,des_x3_3)
inter_des_x33[inter_des_x33>=0.5] = 1
inter_des_x33[inter_des_x33<0.5] = 0
inter_des_x33 = inter_des_x33[:,np.newaxis]

fig, axs = plt.subplots(4,4,figsize=(12,12)) # Creating the subplots mosaic

# Ploting distributions

bandwidth = 0.1 # Parameter for kernel estimation

# For Sepal length
dx0_1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx0_1.fit(x0[0:50, None])
logprob_1 = dx0_1.score_samples(x0_b[:, None])

dx0_2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx0_2.fit(x0[50:100, None])
logprob_2 = dx0_2.score_samples(x0_b[:, None])

dx0_3 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx0_3.fit(x0[100:150, None])
logprob_3 = dx0_3.score_samples(x0_b[:, None])

axs[0,0].fill_between(x0_b, np.exp(logprob_1), color=C_blue,  label= 'Setosa', alpha=0.2)
axs[0,0].fill_between(x0_b, np.exp(logprob_2), color=C_red,  label= 'Versicolor', alpha=0.2)
axs[0,0].fill_between(x0_b, np.exp(logprob_3), color=C_green,  label= 'Virginica', alpha=0.2)
axs[0,0].grid(visible=True, alpha=0.3, linestyle=':')
axs[0,0].set_facecolor(color=C_grayish_back)
axs[0,0].set_xlim(left=x0_b.min(), right=x0_b.max())
axs[0,0].legend()

# For Sepal width
dx1_1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx1_1.fit(x1[0:50, None])
logprob_1 = dx1_1.score_samples(x1_b[:, None])

dx1_2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx1_2.fit(x1[50:100, None])
logprob_2 = dx1_2.score_samples(x1_b[:, None])

dx1_3 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx1_3.fit(x1[100:150, None])
logprob_3 = dx1_3.score_samples(x1_b[:, None])

axs[1,1].fill_between(x1_b, np.exp(logprob_1), color=C_blue,  label= 'Setosa', alpha=0.2)
axs[1,1].fill_between(x1_b, np.exp(logprob_2), color=C_red,  label= 'Versicolor', alpha=0.2)
axs[1,1].fill_between(x1_b, np.exp(logprob_3), color=C_green,  label= 'Virginica', alpha=0.2)
axs[1,1].grid(visible=True, alpha=0.3, linestyle=':')
axs[1,1].set_facecolor(color=C_grayish_back)
axs[1,1].set_xlim(left=x1_b.min(), right=x1_b.max())

# For Petal length
dx2_1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx2_1.fit(x2[0:50, None])
logprob_1 = dx2_1.score_samples(x2_b[:, None])

dx2_2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx2_2.fit(x2[50:100, None])
logprob_2 = dx2_2.score_samples(x2_b[:, None])

dx2_3 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx2_3.fit(x2[100:150, None])
logprob_3 = dx2_3.score_samples(x2_b[:, None])

axs[2,2].fill_between(x2_b, np.exp(logprob_1), color=C_blue,  label= 'Setosa', alpha=0.2)
axs[2,2].fill_between(x2_b, np.exp(logprob_2), color=C_red,  label= 'Versicolor', alpha=0.2)
axs[2,2].fill_between(x2_b, np.exp(logprob_3), color=C_green,  label= 'Virginica', alpha=0.2)
axs[2,2].grid(visible=True, alpha=0.3, linestyle=':')
axs[2,2].set_facecolor(color=C_grayish_back)
axs[2,2].set_xlim(left=x2_b.min(), right=x2_b.max())

# For Petal width
dx3_1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx3_1.fit(x3[0:50, None])
logprob_1 = dx3_1.score_samples(x3_b[:, None])

dx3_2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx3_2.fit(x3[50:100, None])
logprob_2 = dx3_2.score_samples(x3_b[:, None])

dx3_3 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
dx3_3.fit(x3[100:150, None])
logprob_3 = dx3_3.score_samples(x3_b[:, None])

axs[3,3].fill_between(x3_b, np.exp(logprob_1), color=C_blue,  label= 'Setosa', alpha=0.2)
axs[3,3].fill_between(x3_b, np.exp(logprob_2), color=C_red,  label= 'Versicolor', alpha=0.2)
axs[3,3].fill_between(x3_b, np.exp(logprob_3), color=C_green,  label= 'Virginica', alpha=0.2)
axs[3,3].grid(visible=True, alpha=0.3, linestyle=':')
axs[3,3].set_facecolor(color=C_grayish_back)
axs[3,3].set_xlim(left=x3_b.min(), right=x3_b.max())
axs[3,3].set_xlabel('Petal width [cm]', fontsize=14, fontfamily='Baskerville')

# Relation between Sepal size and width
D_01_1 = inter_des_x10@np.transpose(inter_des_x00)
D_01_1 = D_01_1.reshape((n_e*n_e,))
D_01_2 = inter_des_x11@np.transpose(inter_des_x01)
D_01_2 = D_01_2.reshape((n_e*n_e,))
D_01_3 = inter_des_x13@np.transpose(inter_des_x03)
D_01_3 = D_01_3.reshape((n_e*n_e,))

axs[0,1].scatter(x1[0:50],x0[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[0,1].scatter(x1[50:100],x0[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[0,1].scatter(x1[100:150],x0[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[0,1].grid(visible=True, alpha=0.3, linestyle=':')
axs[0,1].set_facecolor(color=C_grayish_back)
axs[0,1].set_ylim(bottom=4, top=8.5)

axs[0,1].scatter(X1m_resh[D_01_1==1], X0m_resh[D_01_1==1], color=C_blue, alpha=alpha)
axs[0,1].scatter(X1m_resh[D_01_2==1], X0m_resh[D_01_2==1], color=C_red, alpha=alpha)
axs[0,1].scatter(X1m_resh[D_01_3==1], X0m_resh[D_01_3==1], color=C_green, alpha=alpha)
axs[0,1].set_ylim(bottom=x0_b.min(), top=x0_b.max())
axs[0,1].set_xlim(left=x1_b.min(), right=x1_b.max())

axs[1,0].scatter(x0[0:50],x1[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[1,0].scatter(x0[50:100],x1[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[1,0].scatter(x0[100:150],x1[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[1,0].grid(visible=True, alpha=0.3, linestyle=':')
axs[1,0].set_facecolor(color=C_grayish_back)
axs[1,0].set_ylabel('Sepal width [cm]', fontsize=14, fontfamily='Baskerville')

axs[1,0].scatter(X0m_resh[D_01_1==1], X1m_resh[D_01_1==1], color=C_blue, alpha=alpha)
axs[1,0].scatter(X0m_resh[D_01_2==1], X1m_resh[D_01_2==1], color=C_red, alpha=alpha)
axs[1,0].scatter(X0m_resh[D_01_3==1], X1m_resh[D_01_3==1], color=C_green, alpha=alpha)
axs[1,0].set_ylim(bottom=x1_b.min(), top=x1_b.max())
axs[1,0].set_xlim(left=x0_b.min(), right=x0_b.max())

# Relation between Sepal size and Petal Size
D_02_1 = inter_des_x20@np.transpose(inter_des_x00)
D_02_1 = D_02_1.reshape((n_e*n_e,))
D_02_2 = inter_des_x21@np.transpose(inter_des_x01)
D_02_2 = D_02_2.reshape((n_e*n_e,))
D_02_3 = inter_des_x23@np.transpose(inter_des_x03)
D_02_3 = D_02_3.reshape((n_e*n_e,))

axs[0,2].scatter(x2[0:50],x0[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[0,2].scatter(x2[50:100],x0[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[0,2].scatter(x2[100:150],x0[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[0,2].grid(visible=True, alpha=0.3, linestyle=':')
axs[0,2].set_facecolor(color=C_grayish_back)

axs[0,2].scatter(X2m_t_resh[D_02_1==1], X0m_resh[D_02_1==1], color=C_blue, alpha=alpha)
axs[0,2].scatter(X2m_t_resh[D_02_2==1], X0m_resh[D_02_2==1], color=C_red, alpha=alpha)
axs[0,2].scatter(X2m_t_resh[D_02_3==1], X0m_resh[D_02_3==1], color=C_green, alpha=alpha)
axs[0,2].set_ylim(bottom=x0_b.min(), top=x0_b.max())
axs[0,2].set_xlim(left=x2_b.min(), right=x2_b.max())

axs[2,0].scatter(x0[0:50],x2[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[2,0].scatter(x0[50:100],x2[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[2,0].scatter(x0[100:150],x2[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[2,0].grid(visible=True, alpha=0.3, linestyle=':')
axs[2,0].set_facecolor(color=C_grayish_back)
axs[2,0].set_ylabel('Petal Length [cm]', fontsize=14, fontfamily='Baskerville')

axs[2,0].scatter(X0m_resh[D_02_1==1], X2m_t_resh[D_02_1==1], color=C_blue, alpha=alpha)
axs[2,0].scatter(X0m_resh[D_02_2==1], X2m_t_resh[D_02_2==1], color=C_red, alpha=alpha)
axs[2,0].scatter(X0m_resh[D_02_3==1], X2m_t_resh[D_02_3==1], color=C_green, alpha=alpha)
axs[2,0].set_ylim(bottom=x2_b.min(), top=x2_b.max())
axs[2,0].set_xlim(left=x0_b.min(), right=x0_b.max())

# Relation between Sepal size and Petal width
D_03_1 = inter_des_x30@np.transpose(inter_des_x00)
D_03_1 = D_03_1.reshape((n_e*n_e,))
D_03_2 = inter_des_x31@np.transpose(inter_des_x01)
D_03_2 = D_03_2.reshape((n_e*n_e,))
D_03_3 = inter_des_x33@np.transpose(inter_des_x03)
D_03_3 = D_03_3.reshape((n_e*n_e,))

axs[0,3].scatter(x3[0:50],x0[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[0,3].scatter(x3[50:100],x0[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[0,3].scatter(x3[100:150],x0[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[0,3].grid(visible=True, alpha=0.3, linestyle=':')
axs[0,3].set_facecolor(color=C_grayish_back)

axs[0,3].scatter(X3m_resh[D_03_1==1], X0m_resh[D_03_1==1], color=C_blue, alpha=alpha)
axs[0,3].scatter(X3m_resh[D_03_2==1], X0m_resh[D_03_2==1], color=C_red, alpha=alpha)
axs[0,3].scatter(X3m_resh[D_03_3==1], X0m_resh[D_03_3==1], color=C_green, alpha=alpha)
axs[0,3].set_ylim(bottom=x0_b.min(), top=x0_b.max())
axs[0,3].set_xlim(left=x3_b.min(), right=x3_b.max())

axs[3,0].scatter(x0[0:50],x3[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[3,0].scatter(x0[50:100],x3[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[3,0].scatter(x0[100:150],x3[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[3,0].grid(visible=True, alpha=0.3, linestyle=':')
axs[3,0].set_facecolor(color=C_grayish_back)
axs[3,0].set_ylabel('Petal width [cm]', fontsize=14, fontfamily='Baskerville')
axs[3,0].set_xlabel('Sepal length [cm]', fontsize=14, fontfamily='Baskerville')

axs[3,0].scatter(X0m_resh[D_03_1==1], X3m_resh[D_03_1==1], color=C_blue, alpha=alpha)
axs[3,0].scatter(X0m_resh[D_03_2==1], X3m_resh[D_03_2==1], color=C_red, alpha=alpha)
axs[3,0].scatter(X0m_resh[D_03_3==1], X3m_resh[D_03_3==1], color=C_green, alpha=alpha)
axs[3,0].set_ylim(bottom=x3_b.min(), top=x3_b.max())
axs[3,0].set_xlim(left=x0_b.min(), right=x0_b.max())

# Relation between Sepal Width and Petal length
D_12_1 = inter_des_x10@np.transpose(inter_des_x20)
D_12_1 = D_12_1.reshape((n_e*n_e,))
D_12_2 = inter_des_x11@np.transpose(inter_des_x21)
D_12_2 = D_12_2.reshape((n_e*n_e,))
D_12_3 = inter_des_x13@np.transpose(inter_des_x23)
D_12_3 = D_12_3.reshape((n_e*n_e,))

axs[1,2].scatter(x2[0:50],x1[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[1,2].scatter(x2[50:100],x1[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[1,2].scatter(x2[100:150],x1[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[1,2].grid(visible=True, alpha=0.3, linestyle=':')
axs[1,2].set_facecolor(color=C_grayish_back)

axs[1,2].scatter(X2m_resh[D_12_1==1], X1m_resh[D_12_1==1], color=C_blue, alpha=alpha)
axs[1,2].scatter(X2m_resh[D_12_2==1], X1m_resh[D_12_2==1], color=C_red, alpha=alpha)
axs[1,2].scatter(X2m_resh[D_12_3==1], X1m_resh[D_12_3==1], color=C_green, alpha=alpha)
axs[1,2].set_ylim(bottom=x1_b.min(), top=x1_b.max())
axs[1,2].set_xlim(left=x2_b.min(), right=x2_b.max())

axs[2,1].scatter(x1[0:50],x2[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[2,1].scatter(x1[50:100],x2[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[2,1].scatter(x1[100:150],x2[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[2,1].grid(visible=True, alpha=0.3, linestyle=':')
axs[2,1].set_facecolor(color=C_grayish_back)

axs[2,1].scatter(X1m_resh[D_12_1==1], X2m_resh[D_12_1==1], color=C_blue, alpha=alpha)
axs[2,1].scatter(X1m_resh[D_12_2==1], X2m_resh[D_12_2==1], color=C_red, alpha=alpha)
axs[2,1].scatter(X1m_resh[D_12_3==1], X2m_resh[D_12_3==1], color=C_green, alpha=alpha)
axs[2,1].set_ylim(bottom=x2_b.min(), top=x2_b.max())
axs[2,1].set_xlim(left=x1_b.min(), right=x1_b.max())

# Relation between Sepal Width and Petal width
D_13_1 = inter_des_x10@np.transpose(inter_des_x30)
D_13_1 = D_13_1.reshape((n_e*n_e,))
D_13_2 = inter_des_x11@np.transpose(inter_des_x31)
D_13_2 = D_13_2.reshape((n_e*n_e,))
D_13_3 = inter_des_x13@np.transpose(inter_des_x33)
D_13_3 = D_13_3.reshape((n_e*n_e,))

axs[1,3].scatter(x3[0:50],x1[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[1,3].scatter(x3[50:100],x1[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[1,3].scatter(x3[100:150],x1[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[1,3].grid(visible=True, alpha=0.3, linestyle=':')
axs[1,3].set_facecolor(color=C_grayish_back)

axs[1,3].scatter(X3m_t_resh[D_13_1==1], X1m_resh[D_13_1==1], color=C_blue, alpha=alpha)
axs[1,3].scatter(X3m_t_resh[D_13_2==1], X1m_resh[D_13_2==1], color=C_red, alpha=alpha)
axs[1,3].scatter(X3m_t_resh[D_13_3==1], X1m_resh[D_13_3==1], color=C_green, alpha=alpha)
axs[1,3].set_ylim(bottom=x1_b.min(), top=x1_b.max())
axs[1,3].set_xlim(left=x3_b.min(), right=x3_b.max())

axs[3,1].scatter(x1[0:50],x3[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[3,1].scatter(x1[50:100],x3[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[3,1].scatter(x1[100:150],x3[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[3,1].grid(visible=True, alpha=0.3, linestyle=':')
axs[3,1].set_facecolor(color=C_grayish_back)
axs[3,1].set_xlabel('Sepal width [cm]', fontsize=14, fontfamily='Baskerville')

axs[3,1].scatter(X1m_resh[D_13_1==1], X3m_t_resh[D_13_1==1], color=C_blue, alpha=alpha)
axs[3,1].scatter(X1m_resh[D_13_2==1], X3m_t_resh[D_13_2==1], color=C_red, alpha=alpha)
axs[3,1].scatter(X1m_resh[D_13_3==1], X3m_t_resh[D_13_3==1], color=C_green, alpha=alpha)
axs[3,1].set_ylim(bottom=x3_b.min(), top=x3_b.max())
axs[3,1].set_xlim(left=x1_b.min(), right=x1_b.max())

# Relation between Petal length and Petal width
D_23_1 = inter_des_x20@np.transpose(inter_des_x30)
D_23_1 = D_23_1.reshape((n_e*n_e,))
D_23_2 = inter_des_x21@np.transpose(inter_des_x31)
D_23_2 = D_23_2.reshape((n_e*n_e,))
D_23_3 = inter_des_x23@np.transpose(inter_des_x33)
D_23_3 = D_23_3.reshape((n_e*n_e,))

axs[2,3].scatter(x3[0:50],x2[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[2,3].scatter(x3[50:100],x2[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[2,3].scatter(x3[100:150],x2[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[2,3].grid(visible=True, alpha=0.3, linestyle=':')
axs[2,3].set_facecolor(color=C_grayish_back)

axs[2,3].scatter(X3m_resh[D_23_1==1], X2m_resh[D_23_1==1], color=C_blue, alpha=alpha)
axs[2,3].scatter(X3m_resh[D_23_2==1], X2m_resh[D_23_2==1], color=C_red, alpha=alpha)
axs[2,3].scatter(X3m_resh[D_23_3==1], X2m_resh[D_23_3==1], color=C_green, alpha=alpha)
axs[2,3].set_ylim(bottom=x2_b.min(), top=x2_b.max())
axs[2,3].set_xlim(left=x3_b.min(), right=x3_b.max())

axs[3,2].scatter(x2[0:50],x3[0:50], color=C_blue, marker = '.', label= 'Setosa')
axs[3,2].scatter(x2[50:100],x3[50:100], color=C_red, marker = '.', label= 'Versicolor')
axs[3,2].scatter(x2[100:150],x3[100:150], color=C_green, marker = '.', label= 'Virginica')
axs[3,2].grid(visible=True, alpha=0.3, linestyle=':')
axs[3,2].set_facecolor(color=C_grayish_back)
axs[3,2].set_xlabel('Petal length [cm]', fontsize=14, fontfamily='Baskerville')

axs[3,2].scatter(X2m_resh[D_23_1==1], X3m_resh[D_23_1==1], color=C_blue, alpha=alpha)
axs[3,2].scatter(X2m_resh[D_23_2==1], X3m_resh[D_23_2==1], color=C_red, alpha=alpha)
axs[3,2].scatter(X2m_resh[D_23_3==1], X3m_resh[D_23_3==1], color=C_green, alpha=alpha)
axs[3,2].set_ylim(bottom=x3_b.min(), top=x3_b.max())
axs[3,2].set_xlim(left=x2_b.min(), right=x2_b.max())

plt.tight_layout()
plt.show()