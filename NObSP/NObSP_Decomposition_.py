import torch
import numpy as np
from kernel_comp import kernel_comp

# Set of functions that implement NObSP in SVM and NN models.
#
# @Copyrigth:  Alexander Caicedo, April 2023

def ObSP(X,Y):

    # Function to compute the oblique projection onto X along the reference space defined by Y.
    # The input data is the following:
    # X: a matriz of size Nxd, containing the basis for the subspace where the data will be projected
    # Y: a matrix of size Nxd, containing the basis for the reference subspace
    # 
    # The function returns the oblique projection matrix of size NxN.

    # Converting the input data into torch tensors

    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    
    if not torch.is_tensor(Y):
        Y = torch.from_numpy(Y)
    
    N, d = X.size() # computing the size of X
        
    P = Y@torch.linalg.pinv(torch.t(Y)@Y)@torch.t(Y) # Conputing the orthogonal projection matriz onto the subsapce given by Y    
    Q = torch.eye(N,N)-P # Computing the complement of P
    
    P_xy = X@torch.linalg.pinv(torch.t(X)@Q@X)@torch.t(X)@torch.t(Q) # Computing the oblique projection matriz onto X along Y
    
    return P_xy

def NObSP_SVM_single(X, Xsv, y_est, model):
    
    # Function to decompose the output of a SVM regression model using oblique subspace projections. The function computes 
    # appropriate kernel matrcies that define the ssubspace of the nonlinear transfromation of the input variables. 
    # These subspaces lie in the same space where the output data is located. This function uses as input the following variables:
    # 
    # X: a matriz of size Nxd, contining the input variables
    # Xsv: a matriz of size N_svxd, contining the support vectors used in athe SVM model
    # y_est: a vector of size Nx1, containing the estimated output for the input data X
    # model: the model for the SVM regression
    # 
    # The function returns d oblique projection matrices of size NxN, the estimated contribution of each input variable on the output,
    # and the alpha coefificents for the out-of-sample extension
    
    N = np.size(X,0) # computing the size of X along dimension 0
    d = np.size(X,1) # computing the size of X along dimension 1
    N_sv = np.size(Xsv,0) # computing the size of X along dimension 0
    y_est = y_est.reshape((-1,1)) # Converting to a vector with dimension 1
    P_xy = np.zeros((N,N,d)) # Initializing proyection matrices
    y_e = np.zeros((N,d)) # Initializing Matriz where the estimated nonlinear contributions will be stored
    Alpha = np.zeros((N_sv,d)) # Initializing the matrix for th Alpha coefficients, out-of-sample extension

    
    Kt = kernel_comp(X,Xsv,model)  # Coputing the kernel matrix for the data X, uing the support vectors
    
    # Main loop for NObSP
    for i in range(d):
        # Defining the input matrix that will be used to find the subspace of the nonlinear transformation of 
        # the input variable x_i, onto which the output will be projected
        X_target = np.zeros((N,d))
        X_target[:,i] = X[:,i]
        
        # Defining the input matrix that will be used to find the reference subspace, along which the data 
        # will be projected.
        X_reference = np.copy(X)
        X_reference[:,i] = 0
        
        # computing the kernel matrices using X_target and X_reference, which will define a basis for each subspace
        Kx_target = kernel_comp(X_target,Xsv,model) # Basis for the subspace of the nonlinear transformation of x_i
        Kx_reference = kernel_comp(X_reference,Xsv,model) # Basis for the  reference subspace of the nonlinear transformation of all variables except x_i
        
        # Computing the oblique projection onto the susbspace defined by the nonlienar transformation of x_i along 
        # the reference subspace, which contains the nonlinear transofrmation of all variables except x_i
        P_xy[:,:,i] = ObSP(Kx_target,Kx_reference) 
        
        y_e[:,[i]] = P_xy[:,:,i]@(y_est-y_est.mean()) # Using the projection matrices to ptoject the output vector and find the nonlinear contribution of each variable.
        
    Alpha = np.linalg.lstsq(Kt, y_e, rcond=None)[0] # Alpha coeficients for the out-of-sample extension
        
    return P_xy, y_e, Alpha

def NObSP_SVM_2order(X, Xsv, y_est, P, model):
    
    # Function to decompose the output of a SVM regression model using oblique subspace projections and considering second order interaction effects. The function computes 
    # appropriate kernel matrcies that define the subspace of the nonlinear transfromation of the input variables. 
    # These subspaces lie in the same space where the output data is located. This function uses as input the following variables:
    # 
    # X: a matriz of size Nxd, contining the input variables
    # Xsv: a matriz of size N_svxd, contining the support vectors used in athe SVM model
    # y_est: a vector of size Nx1, containing the estimated output for the input data X
    # P: A tensor of size NxNxd containing the projection matrices for the first order interactions.
    # model: the model for the SVM regression
    # 
    # The function returns d oblique projection matrices of size NxN, the estimated contribution of each input variable on the output.
    
    N = np.size(X,0) # computing the size of X along dimension 0
    d = np.size(X,1) # computing the size of X along dimension 1
    d_com = np.int_(d*(d-1)/2) # Number of interaction effects to compute
    y_est = y_est.reshape((-1,1)) # Converting to a vector with dimension 1
    P_xy = np.zeros((N,N,d_com)) # Initializing proyection matrices
    y_e = np.zeros((N,d_com)) # Initializing Matriz where the estimated nonlinear contributions will be stored
    

    for i in range(d):
        for k in range(i+1,d):
            index_aux = np.int_(k-(i+1)+i*d-np.sum(range(0,i+1))) # Converting the variables i and k into a linear index
            
            # Defining the input matrix that will be used to find the subspace of the nonlinear transformation of 
            # the input variables x_i and x_k, onto which the output will be projected
            X_target = np.zeros((N,d))
            X_target[:,i] = X[:,i]
            X_target[:,k] = X[:,k]
            
            # Defining the input matrix that will be used to find the reference subspace, along which the data 
            # will be projected.
            X_reference = np.copy(X)
            X_reference[:,i] = 0
            X_reference[:,k] = 0
            
            # computing the kernel matrices using X_target and X_reference, which will define a basis for each subspace
            Kx_target = kernel_comp(X_target,Xsv,model)  # Basis for the subspace of the nonlinear transformation of x_i and x_k
            Kx_reference = kernel_comp(X_reference,Xsv,model) # Basis for the  reference subspace of the nonlinear transformation of all variables except x_i and x_k
            
            # Computing the oblique projection onto the susbspace defined by the nonlinear transformation of x_i and x_k along 
            # the reference subspace, which contains the nonlinear transofrmation of all variables except x_i and x_k
            P_xy[:,:,index_aux] = ObSP(Kx_target,Kx_reference)
            
            y_e[:,[index_aux]] = P_xy[:,:,index_aux]@(y_est-y_est.mean()) # Using the projection matrices to project the output vector and find the nonlinear interaction effect of each pair of variables.
            
    return P_xy, y_e
