import torch
from torch import nn
import torch.nn.functional as F

# Class to create a classification model using feed-forward neural networks
#
# @Copyrigth:  Alexander Caicedo, April 2023

class Classifier_NN(nn.Module):
    def __init__(self, in_number, out_number):
        super(Classifier_NN, self).__init__()
        self.Linear_1 = nn.Linear(in_features = in_number, out_features = 100)
        self.Linear_2 = nn.Linear(in_features = 100, out_features = 500)
        self.Linear_3 = nn.Linear(in_features = 500, out_features = out_number)
        
    def forward(self,x):
        x = F.relu(self.Linear_1(x))
        x_t= F.relu(self.Linear_2(x)) # Computing the transformation done to the vector in the layer previous to the output
        y_lin = self.Linear_3(x_t)
        x = F.softmax(y_lin,dim=1)
        return x, x_t, y_lin
    
class Classifier_CIFAR(nn.Module):
    def __init__(self, in_number):
        super(Classifier_CIFAR, self).__init__()
        self.Linear_1 = nn.Linear(in_features = in_number, out_features = 512)
        self.Linear_2 = nn.Linear(in_features = 512, out_features = 256)
        self.Linear_3 = nn.Linear(in_features = 256, out_features = 128)
        self.Linear_4 = nn.Linear(in_features = 128, out_features = 64)
        self.Linear_5 = nn.Linear(in_features = 64, out_features = 10)
        
    def forward(self,x):
        x = F.relu(self.Linear_1(x))
        x = F.relu(self.Linear_2(x))
        x = F.relu(self.Linear_3(x))
        x_t= F.relu(self.Linear_4(x)) # Computing the transformation done to the vector in the layer previous to the output
        y_lin = self.Linear_5(x_t)
        x = F.softmax(y_lin,dim=1)
        return x, x_t, y_lin