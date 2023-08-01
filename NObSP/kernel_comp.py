import numpy as np
from sklearn.metrics.pairwise import linear_kernel,polynomial_kernel,rbf_kernel,laplacian_kernel,sigmoid_kernel,chi2_kernel

def kernel_comp(Xt,Xsv,model):
    # Funtion to compute the jkernel matrices between elements in matrix Xt and Xsv,. The kernel funciton is defined by the 
    # function specified in the model. The funtion uses as input the following variables:
    #
    # Xt: A matrix of size Nxd containing information N observations of a d-dimensional space
    # Xsv: A matrix comntaining th esupport vectors of the model
    # model: and SVM model
    
    d = np.size(Xt,1) # computing the size of X along dimension 1
    
    # Computing the different kernels
    if model.kernel=='linear':
        kernel = linear_kernel(Xt, Xsv)
    elif model.kernel=='polynomial':
        kernel = polynomial_kernel(Xt, Xsv, degree = model.degree, gamma = 1 / (d * Xt.var()), coef0 = model.coef0)
    elif model.kernel=='rbf':
        kernel = rbf_kernel(Xt, Xsv, gamma = 1 / (d * Xt.var()))
    elif model.kernel=='laplacian':
        kernel = laplacian_kernel(Xt, Xsv, gamma = 1 / (d * Xt.var()))
    elif model.kernel=='sigmoidal':
        kernel = sigmoidal_kernel(Xt, Xsv, gamma = 1 / (d * Xt.var()), coef0 = model.coef0)
    else: 
        kernel = chi2_kernel(Xt, Xtest)
    return kernel