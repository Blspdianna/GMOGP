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
FilePath1 = "/home/home1/student/yijue/MTGPdai/data/KUKAtrain.mat"
FilePath2 = "/home/home1/student/yijue/MTGPdai/data/KUKAtest.mat"
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


######################## JURA ######################
# FilePath = "/home/home1/student/yijue/MTGPdai/jura.mat"
# Data = loadmat(FilePath, squeeze_me=True, struct_as_record=False, mat_dtype=True)
#
# train_x = torch.from_numpy(Data['X_train']).float()
# train_y = torch.from_numpy(Data['Y_train']).float()
#
# train_y0 = train_y[:, 0]
# train_y1 = train_y[:, 1]
# train_y2 = train_y[:, 2]
#
# train_x0=train_x1=train_x2=train_x
#
# test_x = torch.from_numpy(Data['X_test']).float()
# test_y = torch.from_numpy(Data['Y_test']).float()
#
#
# test_y0 = test_y[:, 0]
# test_y1 = test_y[:, 1]
# test_y2 = test_y[:, 2]
# leng = train_y.shape[1]

# ####################### ECG fetal ########################
# file = "/home/home1/student/yijue/MTGPdai/ECG_data/r01.edf"
# data = mne.io.read_raw_edf(file)
# raw_data = data.get_data()
# # you can get the metadata included in the file and a list of all channels:
# info = data.info
# channels = data.ch_names
#
#
# Samples = torch.from_numpy(raw_data)[:,1:8001]
# x = Samples.float()
# x = x.t()-x.min(1)[0]
# x = 2 * (x / x.max(0)[0]) - 1
#
# leng = 5
# train_x = torch.arange(1,6001).unsqueeze(1).float()
# train_y0 = x[0:6000,0]
# train_y1 = x[0:6000,1]
# train_y2 = x[0:6000,2]
# train_y3 = x[0:6000,3]
# train_y4 = x[0:6000,4]
#
# test_x = torch.arange(6001,8001).unsqueeze(1).float()
# test_y0 = x[6000:8000,0]
# test_y1 = x[6000:8000,1]
# test_y2 = x[6000:8000,2]
# test_y3 = x[6000:8000,3]
# test_y4 = x[6000:8000,4]
#

####################### Traffic Data ########################
# Data_cell0 = pd.read_csv("/home/home1/student/yijue/MTGPdai/Traffic_data/32beam_new20210121_traffic_data_cell_0.csv")
# samples0 = Data_cell0.values
# x0 = torch.from_numpy(samples0).float()
# x0 = (x0 / x0.max(0)[0])+0.1
# Data_cell1 = pd.read_csv("/home/home1/student/yijue/MTGPdai/Traffic_data/32beam_new20210121_traffic_data_cell_1.csv")
# samples1 = Data_cell1.values
# x1 = torch.from_numpy(samples1).float()
# x1 = (x1 / x1.max(0)[0])+0.1
# Data_cell2 = pd.read_csv("/home/home1/student/yijue/MTGPdai/Traffic_data/32beam_new20210121_traffic_data_cell_2.csv")
# samples2 = Data_cell2.values
# x2 = torch.from_numpy(samples2).float()
# x2 = (x2 / x2.max(0)[0])+0.1
#
# train_x = torch.arange(0,500).unsqueeze(1).float()
# train_y0 = x0[0:500,2]
# train_y1 = x0[0:500,1]
# train_y2 = x1[0:500,2]
# train_y3 = x1[0:500,1]
# train_y4 = x2[0:500,2]
# train_y5 = x2[0:500,1]
#
# test_x = torch.arange(500,672).unsqueeze(1).float()
# test_y0 = x0[500:,2]
# test_y1 = x0[500:,1]
# test_y2 = x1[500:,2]
# test_y3 = x1[500:,1]
# test_y4 = x2[500:,2]
# test_y5 = x2[500:,1]
# leng = 6


#################### EEG #######################
# train = pd.read_csv("/home/home1/student/yijue/MTGPdai/data/EEG_train.csv", index_col=0)
# test = pd.read_csv("/home/home1/student/yijue/MTGPdai/data/EEG_test.csv", index_col=0)
# all = pd.read_csv("/home/home1/student/yijue/MTGPdai/data/EEG_all.csv", index_col=0)
# train_x_neat = torch.tensor(train.index).float()
#
# torch.manual_seed(110)
# ind = list(range(256))
# np.random.shuffle(ind)
# train_x = train_x_neat[ind]
#
#
# train_x = train_x[0:156].unsqueeze(1)
# train_y0 = torch.tensor(all['FZ'].values[ind][0:156]).float()
# train_y1 = torch.tensor(all['F1'].values[ind][0:156]).float()
# train_y2 = torch.tensor(all['F2'].values[ind][0:156]).float()
# train_y3 = torch.tensor(all['F3'].values[ind][0:156]).float()
# train_y4 = torch.tensor(all['F4'].values[ind][0:156]).float()
# train_y5 = torch.tensor(all['F5'].values[ind][0:156]).float()
# train_y6 = torch.tensor(all['F6'].values[ind][0:156]).float()
#
#
# test_x = train_x_neat[ind][156:256].unsqueeze(1)
# test_y1 = torch.tensor(all['F1'].values[ind][156:256]).float()
# test_y2 = torch.tensor(all['F2'].values[ind][156:256]).float()
# test_y0 = torch.tensor(all['FZ'].values[ind][156:256]).float()
# test_y3 = torch.tensor(all['F3'].values[ind][156:256]).float()
# test_y4 = torch.tensor(all['F4'].values[ind][156:256]).float()
# test_y5 = torch.tensor(all['F5'].values[ind][156:256]).float()
# test_y6 = torch.tensor(all['F6'].values[ind][156:256]).float()
# leng = 7

#################### SARCOS #######################
# FilePath = "/home/home1/student/yijue/MTGPdai/data/sarcos_inv.mat"
# Data_train = loadmat(FilePath, squeeze_me=True, struct_as_record=False, mat_dtype=True)
# path = "/home/home1/student/yijue/MTGPdai/data/sarcos_inv_test.mat"
# Data_test = loadmat(path, squeeze_me=True, struct_as_record=False, mat_dtype=True)
# train = torch.FloatTensor(Data_train['sarcos_inv'])
# n = 20000
# train_x = train[0:n, 0:21]
# train_y0 = train[0:n, 22]
# train_y1 = train[0:n, 23]
# train_y2 = train[0:n, 24]
# train_y3 = train[0:n, 27]
#
# test = torch.FloatTensor(Data_test['sarcos_inv_test'])
# test_x = test[:, 0:21]
# test_y0 = test[:, 22]
# test_y1 = test[:, 23]
# test_y2 = test[:, 24]
# test_y3 = test[:, 27]
# leng = 4


# ####################### SYTHETIC DATA #################
# leng=5
# torch.manual_seed(110)
# x = torch.rand(3000, 2)
# x = x - x.min(0)[0]
# x = 2 * (x / x.max(0)[0]) - 1
#
#
# train_n = int((0.6*len(x)))
# k = int((0.2*len(x)))
# ind = list(range(len(x[0:train_n+k])))
# np.random.shuffle(ind)
# split = ind[0:train_n]
# lsplit = ind[train_n:]
#
# torch.manual_seed(110)
# tx = torch.t(x)
# err = torch.randn(5, x.size(0))*0.2 #var = 0.04
# y0 = (err[0] + 2*torch.cos(tx[0]+tx[1]))
# y1 = (err[1] + 2*torch.cos(tx[0]+tx[1]) + (tx[0]+tx[1]).pow(2))
# y2 = torch.sinh(2*torch.arcsinh(2*torch.cos(tx[0]+tx[1]) + (tx[0]+tx[1]).pow(2)) + err[2])
# y3 = 3*torch.tanh(torch.exp(1+tx[0]*tx[1])*torch.log(tx[0]+3) + 2*torch.cos(tx[0]+tx[1]) + err[3])
# y4 = 5*torch.exp(1+tx[0]*tx[1])*torch.log(tx[0]+3) + err[4]
#
# # datarange = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,
# #              220,240,260,280,300,320,340,360,380,400,440,480,520,560,600,700,800,900,1000,1100]
# # res = []
# # for ss in datarange:
# ss = 1800
#
# train_x = x[split][0:ss]
# valid_x = x[lsplit]
# test_x = x[train_n+k:, :]
#
# train_y0 = y0[split][0:ss]
# valid_y0 = y0[lsplit]
# test_y0 = y0[train_n+k:]
#
# train_y1 = y1[split][0:ss]
# valid_y1 = y1[lsplit]
# test_y1 = y1[train_n+k:]
#
# train_y2 = y2[split][0:ss]
# valid_y2 = y2[lsplit]
# test_y2 = y2[train_n+k:]
#
# train_y3 = y3[split][0:ss]
# valid_y3 = y3[lsplit]
# test_y3 = y3[train_n+k:]
#
# train_y4 = y4[split][0:ss]
# valid_y4 = y4[lsplit]
# test_y4 = y4[train_n+k:]
####################################################

data_dim = train_x.size(-1)
print(train_x.shape)

# class LargeFeatureExtractor(torch.nn.Sequential):
#     def __init__(self):
#         super(LargeFeatureExtractor, self).__init__()
#         self.add_module('linear1', torch.nn.Linear(data_dim, 100))
#         self.add_module('relu1', torch.nn.ReLU())
#         self.add_module('linear2', torch.nn.Linear(100, 50))
#         self.add_module('relu2', torch.nn.ReLU())
#         self.add_module('linear3', torch.nn.Linear(50, 5))
#         self.add_module('relu3', torch.nn.ReLU())
#         self.add_module('linear4', torch.nn.Linear(5, 2))

# for i in range(leng):
#     exec('feature_extractor'+str(i)+' = LargeFeatureExtractor()')


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
        # self.covar_module = Deepkernel()

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    # x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    # print(exp_x)
    exp_x = exp_x - torch.eye(leng) * torch.diag(exp_x)
    # print(exp_x)
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
        # print("x", x)

        # print(self.W1)
        e = torch.matmul(x, x.t())
        # print(e)
        e = self.leakyrelu(torch.mm(e, self.W1)+self.B)
        # attention = F.softmax(e, dim=1)
        attention = softmax_one(e, dim=1)
        # print(attention)
        return attention


# v0 = np.linalg.norm(train_y0[0:156])
# v1 = np.linalg.norm(train_y1[0:156])
# v2 = np.linalg.norm(train_y2[0:156])
# v3 = np.linalg.norm(train_y3[0:156])
# v4 = np.linalg.norm(train_y4[0:156])
# v5 = np.linalg.norm(train_y5[0:156])
# v6 = np.linalg.norm(train_y6[0:156])


v0 = np.linalg.norm(train_y0)
v1 = np.linalg.norm(train_y1)
v2 = np.linalg.norm(train_y2)
v3 = np.linalg.norm(train_y3)
v4 = np.linalg.norm(train_y4)
v5 = np.linalg.norm(train_y5)
v6 = np.linalg.norm(train_y6)



# features = torch.cat(((train_y0[0:156]/v0).unsqueeze(0), (train_y1[0:156]/v1).unsqueeze(0), (train_y2[0:156]/v2).unsqueeze(0), (train_y3[0:156]/v3).unsqueeze(0)
#                         , (train_y4[0:156]/v4).unsqueeze(0), (train_y5[0:156]/v5).unsqueeze(0), (train_y6[0:156]/v6).unsqueeze(0)), 0)


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

        # exec('self.feature_extractor = feature_extractor'+str(ind)+'')
        self.id = ind

    def forward(self, x):
        # projected_x = self.feature_extractor(x)
        # projected_x = projected_x - projected_x.min(0)[0]
        # projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1
        # x = projected_x
        #### here can not write the exec code, the performance is weird, it may cause the code not see the parameters.
        ##### the exec code can not write in the forward param.

        a_coef = self.WGat(features)
        a_coef[self.id][self.id] = 1
        # print(a_coef[self.id])
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

model0.to('cuda:0')
model1.to('cuda:1')
model2.to('cuda:0')
model3.to('cuda:1')
model4.to('cuda:0')
model5.to('cuda:1')
model6.to('cuda:0')


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
    # print(model.models[0].feature_extractor.state_dict())
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
    test_y_list = [test_y0, test_y1, test_y2, test_y3, test_y4]
    testnll = - mll(predictions, test_y_list)
    print("TEST NLL", testnll)
#     mean0 = predictions[0].mean*Data['Y_std'][0]+Data['Y_mean'][0]
#     mean1 = predictions[1].mean*Data['Y_std'][1]+Data['Y_mean'][1]
#     mean2 = predictions[2].mean * Data['Y_std'][2] + Data['Y_mean'][2]
#     Y_pred = torch.cat((mean0.unsqueeze(1), mean1.unsqueeze(1), mean2.unsqueeze(1)), dim=1).numpy()
# #
# # #
# r0 = np.sqrt(np.mean(np.square(Y_pred - Data['Y_test_ground']))) / np.std(Data['Y_test_ground'])
# print(r0)

# r1 = np.mean(np.abs(Y_pred - Data['Y_test_ground'])) / np.std(Data['Y_test_ground'])

    # lower, upper = predictions[0].confidence_region()
    # a = test_x.squeeze().tolist()
    # indies = np.argsort(a, kind='heapsort')
    #
    # f, ax = plt.subplots(1, 1, figsize=(10, 3))
    #
    # # Get upper and lower confidence bounds
    # # Plot training data as black stars
    # ax.scatter(train_x.numpy(), train_y0.numpy(), marker='x', color='gray',label = 'Training')
    # # Plot predictive means as blue line
    # ax.plot(test_x[indies].numpy(), test_y0[indies].numpy(), 'kv', markerfacecolor='none',label = 'Testing')
    # ax.fill_between(test_x[indies].squeeze().numpy(), lower[indies].numpy(), upper[indies].numpy(), alpha=0.5,color = 'darkorange')
    # ax.plot(test_x[indies].numpy(), predictions[0].mean[indies], color='brown',label = 'G-GPS')
    # plt.legend(loc='lower left')
    # plt.grid(linestyle = '--')
    # plt.savefig('EEG_GGPS.pdf', format='pdf')
    # plt.show()
#
# mse0 = ((predictions[0].mean - test_y0)**2).mean()
# std0 = (((test_y0.mean() - test_y0)**2).mean())
# smse0 = mse0/std0
#
#
# mse1 = ((predictions[1].mean - test_y1)**2).mean()
# std1 = (((test_y1.mean() - test_y1)**2).mean())
# smse1 = mse1/std1
#
#
# mse2 = ((predictions[2].mean - test_y2)**2).mean()
# std2 = (((test_y2.mean() - test_y2)**2).mean())
# smse2 = mse2/std2
#
#
# mse3 = ((predictions[3].mean - test_y3)**2).mean()
# std3 = (((test_y3.mean() - test_y3)**2).mean())
# smse3 = mse3/std3
# #
# # #
# mse4 =((predictions[4].mean - test_y4)**2).mean()
# std4 = (((test_y4.mean() - test_y4)**2).mean())
# smse4 = mse4/std4
# #
# # mse5 = ((predictions[5].mean - test_y5)**2).mean()
# # std5 = (((test_y5.mean() - test_y5)**2).mean())
# # smse5 = mse5/std5
# #
# # mse6 = ((predictions[6].mean - test_y6)**2).mean()
# # std6 = (((test_y6.mean() - test_y6)**2).mean())
# # smse6 = mse6/std6
# # # # #
# # # # #
# print((smse1+smse2+smse0+smse3)/4)

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


