import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from hdf5storage import loadmat
import pandas as pd
import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn
import mne



####################### KUKA ######################
FilePath1 = "/home/data/KUKAtrain.mat"
FilePath2 = "/home/data/KUKAtest.mat"
trainData = loadmat(FilePath1, squeeze_me=True, struct_as_record=False, mat_dtype=True)
testData = loadmat(FilePath2, squeeze_me=True, struct_as_record=False, mat_dtype=True)

train = torch.from_numpy(trainData['Train'])
test = torch.from_numpy(testData['Test'])

train_x = train[:, 0:21]
train_y0 = train[:, 21]
train_y1 = train[:, 22]
train_y2 = train[:, 23]
train_y3 = train[:, 24]
train_y4 = train[:, 25]
train_y5 = train[:, 26]
train_y6 = train[:, 27]



test_x = test[:, 0:21]
test_y0 = test[:, 21]
test_y1 = test[:, 22]
test_y2 = test[:, 23]
test_y3 = test[:, 24]
test_y4 = test[:, 25]
test_y5 = test[:, 26]
test_y6 = test[:, 27]
leng = 7


#####################MAIN CODE###############

data_dim = train_x.size(-1)
print(train_x.shape)


class Deepkernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def forward(self, x1, x2, **params):
        covres = torch.mm(x1, torch.t(x2))
        return covres

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #compute exponentials
    exp_x = torch.exp(x)
    exp_x = exp_x - torch.eye(leng) * torch.diag(exp_x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.W1 = nn.Parameter(torch.ones(size=(leng, leng)))
        self.B = nn.Parameter(torch.randn(size=(leng, leng)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x):
        e = torch.matmul(x, x.t())
        e = self.leakyrelu(torch.mm(e, self.W1)+self.B)
        attention = softmax_one(e, dim=1)
        return attention


v0 = np.linalg.norm(train_y0)
v1 = np.linalg.norm(train_y1)
v2 = np.linalg.norm(train_y2)
v3 = np.linalg.norm(train_y3)
v4 = np.linalg.norm(train_y4)
v5 = np.linalg.norm(train_y5)
v6 = np.linalg.norm(train_y6)



features = torch.cat(((train_y0/v0).unsqueeze(0), (train_y1/v1).unsqueeze(0), (train_y2/v2).unsqueeze(0), (train_y3/v3).unsqueeze(0),
                      (train_y4/v4).unsqueeze(0),(train_y5/v5).unsqueeze(0),(train_y6/v6).unsqueeze(0)), 0)


for i in range(leng):
    exec('Gatnet'+str(i)+' = GAT(nfeat=features.shape[1], nhid=10, alpha=0.2)')


class LMCGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ind):
        super().__init__(train_x, train_y, likelihood)

        for i in range(leng):
            exec('self.covar'+str(i)+' = model'+str(i)+'.covar_module.cpu()')
            exec('self.mean'+str(i)+' = model'+str(i)+'.mean_module.cpu()')
            exec('self.WGat = Gatnet' + str(ind) + '')

        self.id = ind

    def forward(self, x):
        #### here can not write the exec code, the performance is weird, it may cause the code not see the parameters.
        ##### the exec code can not write in the forward param.

        a_coef = self.WGat(features)
        a_coef[self.id][self.id] = 1

        mean_x = a_coef[self.id][0] * self.mean0(x) + \
                 a_coef[self.id][1] * self.mean1(x) + \
                 a_coef[self.id][2] * self.mean2(x)+ \
                 a_coef[self.id][3] * self.mean3(x)+ \
                 a_coef[self.id][4] * self.mean4(x)+ \
                 a_coef[self.id][5] * self.mean5(x)+ \
                 a_coef[self.id][6] * self.mean6(x)



        covar_x = a_coef[self.id][0].pow(2)*self.covar0(x) + \
                  a_coef[self.id][1].pow(2) * self.covar1(x) + \
                  a_coef[self.id][2].pow(2) * self.covar2(x)+ \
                  a_coef[self.id][3].pow(2) * self.covar3(x)+ \
                  a_coef[self.id][4].pow(2) * self.covar4(x)+ \
                  a_coef[self.id][5].pow(2) * self.covar5(x)+ \
                  a_coef[self.id][6].pow(2) * self.covar6(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


for i in range(leng):
    exec('likelihood'+str(i)+' = gpytorch.likelihoods.GaussianLikelihood()')
    exec('model'+str(i)+' = ExactGPModel(train_x, train_y'+str(i)+', likelihood'+str(i)+')')




for i in range(leng):
    exec('likelihood_lmc'+str(i)+' = gpytorch.likelihoods.GaussianLikelihood()')
    exec('model_LMC' + str(i) + ' = LMCGPModel(train_x, train_y' + str(i) + ', likelihood_lmc' + str(
        i) + ',ind = ' + str(i) + ')')

model = gpytorch.models.IndependentModelList(model_LMC0)
likelihood = gpytorch.likelihoods.LikelihoodList(model_LMC0.likelihood)
for i in range(leng):
    if i != 0:
        exec('model.models.append(model_LMC'+str(i)+')')
        exec('likelihood.likelihoods.append(model_LMC'+str(i)+'.likelihood)')

from gpytorch.mlls import SumMarginalLogLikelihood

mll = SumMarginalLogLikelihood(likelihood, model)

training_iterations = 80


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the Adam optimizer # Includes GaussianLikelihood parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.08,
                             weight_decay=5e-4)  # Includes GaussianLikelihood parameters
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

time_start = time.time()


for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(*model.train_inputs)
    loss = -mll(output, model.train_targets)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
    scheduler.step()

time_end = time.time()
Time = (time_end - time_start)
print(Time/training_iterations)
# Set into eval mode
model.eval()
likelihood.eval()


# Make predictions (use the same test points)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # This contains predictions for both outcomes as a list
    predictions = likelihood(*model(test_x, test_x,test_x,test_x,test_x,test_x,test_x))


###################### RMSE ###################################
mse0 = np.sqrt(((predictions[0].mean - test_y0)**2).mean())

mse1 = np.sqrt(((predictions[1].mean - test_y1)**2).mean())

mse2 = np.sqrt(((predictions[2].mean - test_y2)**2).mean())

mse3 = np.sqrt(((predictions[3].mean - test_y3)**2).mean())

mse4 = np.sqrt(((predictions[4].mean - test_y4)**2).mean())
mse5 = np.sqrt(((predictions[5].mean - test_y5)**2).mean())

mse6 = np.sqrt(((predictions[6].mean - test_y6)**2).mean())

mean_rmse = (mse0 + mse1 + mse2 + mse3+ mse4+ mse5+ mse6) / 7

print(mean_rmse)


