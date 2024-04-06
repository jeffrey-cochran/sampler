#!/usr/bin/env python
# coding: utf-8

# In[10]:

# In[1]:


"""
This cell imports required library and data files.
"""
# Import libraries
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
#from torchsummary import summary
import torch_rbf as rbf
#from rbf_layer import RBFLayer, l_norm, rbf_gaussian

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 1008)
np.random.seed(hash("improves reproducibility") % 1008)
torch.manual_seed(hash("by removing stochasticity") % 1008)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 1008)

# Device configuration
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

# Load data (change this path to the path of the folder in which your data is stored)
get_ipython().run_line_magic('cd', './sparse_data/sensorgrid12x12_Nf144_Nk144_Nh144_Ntemp268')
#%cd ./sparse_data/sensorgrid10x10_Nf100_Nk100_Ntemp224
get_ipython().system('ls')

f_branch_data = np.load("f_branch_data.npy")
k_branch_data = np.load("k_branch_data.npy")
h_branch_data = np.load("h_branch_data.npy")
trunk_data = np.load("trunk_data.npy")
output_data = np.load("output_data.npy")
f_idx = np.load("f_idx.npy")
k_idx = np.load("k_idx.npy")
h_idx = np.load("h_idx.npy")
f_fullfield = np.load("./../../fullfield_data/f_fullfield.npy")
k_fullfield = np.load("./../../fullfield_data/k_fullfield.npy")
h_fullfield = np.load("./../../fullfield_data/h_fullfield.npy")
u_fullfield = np.load("./../../fullfield_data/u_fullfield.npy")

# =================================================
n_temp_sensor = 268#224       # no. of temp. sensors in training data
n_train = 1000           # no. of training samples
n_val = 1000              # no. of validation samples
n_val_start = 9000        # no. of samples after which in the combined dataset the validation set starts (by default it should be n_train, but we are specifically 
                          # defining it here to consider the cases with lean training i.e. smaller training and/or validation sets)
# ==================== train ======================
f_branch_train_np = f_branch_data[:n_temp_sensor*n_train, :]
k_branch_train_np = k_branch_data[:n_temp_sensor*n_train, :]
h_branch_train_np = h_branch_data[:n_temp_sensor*n_train, :]
trunk_train_np = trunk_data[:n_temp_sensor*n_train, :]
output_train_np = output_data[:n_temp_sensor*n_train, :]
# ================== validation ===================
Nx = Ny = 32
xvec_domain, yvec_domain = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
# sensor node values
f_at_sensor = f_fullfield[n_val_start:n_val_start+n_val, f_idx[:, 0], f_idx[:, 1]]
k_at_sensor = k_fullfield[n_val_start:n_val_start+n_val, k_idx[:, 0], k_idx[:, 1]]
h_at_sensor = h_fullfield[n_val_start:n_val_start+n_val, h_idx[:, 0], h_idx[:, 1]]
#h_at_sensor = h_fullfield_1d[:, h_idx]

f_branch_val_np = np.repeat(f_at_sensor, repeats=32*32, axis=0)
k_branch_val_np = np.repeat(k_at_sensor, repeats=32*32, axis=0)
h_branch_val_np = np.repeat(h_at_sensor, repeats=32*32, axis=0)

trunk_val_np = np.tile(np.array([xvec_domain.flatten(), yvec_domain.flatten()]).T, (n_val, 1))
output_val_2d = u_fullfield[n_val_start:n_val_start+n_val, :, :].reshape([-1, 32*32])
output_val_np = np.expand_dims(output_val_2d.flatten(), axis=1)

# Print out basic information about different arrays
def log_info(array, array_name):
  print(f"{array_name}: shape={array.shape}, max={array.max()}, min={array.min()}")

print("==============")
log_info(f_branch_data, "f_branch_data")
log_info(k_branch_data, "k_branch_data")
log_info(h_branch_data, "h_branch_data")
log_info(trunk_data, "trunk_data")
log_info(output_data, "output_data")
print("==============")
log_info(f_branch_train_np, "f_branch_train_np")
log_info(k_branch_train_np, "k_branch_train_np")
log_info(h_branch_train_np, "h_branch_train_np")
log_info(trunk_train_np, "trunk_train_np")
log_info(output_train_np, "output_train_np")
print("==============")
log_info(f_branch_val_np, "f_branch_val_np")
log_info(k_branch_val_np, "k_branch_val_np")
log_info(h_branch_val_np, "h_branch_val_np")
log_info(trunk_val_np, "trunk_val_np")
log_info(output_val_np, "output_val_np")
print("==============")
log_info(f_branch_data, "f_branch_data")
log_info(k_branch_data, "k_branch_data")
log_info(h_branch_data, "h_branch_data")
log_info(trunk_data, "trunk_data")
log_info(output_data, "output_data")

# In[2]:

"""
This cell imports wandb library
You first might need to create a wandb account at https://www.wandb.ai and then replace the following key
"""
import wandb
#wandb.login(key="1dab7e5b63fb3ada6629beb878092d701cdff795")   # replace this key with the one from your account.
wandb.login(key="0c0c05acffed23b17efaf91e91b3acd6b84203c9")   # replace this key with the one from your account.


# In[4]:


"""
PyTorch dataloader for different streams of data
"""
class NumpyToPytorchDataset(torch.utils.data.Dataset):
  def __init__(self, f_branch, k_branch, h_branch, trunk, output, device, batch_size):

    self.f_branch_torch = torch.tensor(f_branch, dtype=torch.float).to(device)
    self.k_branch_torch = torch.tensor(k_branch.reshape([1, k_branch.shape[0], 12, 12]), dtype=torch.float).to(device)
    self.h_branch_torch = torch.tensor(h_branch, dtype=torch.float).to(device)
    self.trunk_torch = torch.tensor(trunk, dtype=torch.float).to(device)
    self.output_torch = torch.tensor(output, dtype=torch.float).to(device).squeeze()
    self.batch_size = batch_size

  def __len__(self):
    return len(self.output_torch)

  def __getitem__(self, idx):
    return self.f_branch_torch[idx], self.k_branch_torch[:, idx, :, :], self.h_branch_torch[idx], self.trunk_torch[idx], self.output_torch[idx]

  
def create_dataloader(f_branch__np, k_branch__np, h_branch__np, trunk__np, output__np, device, batch_size, if_shuffle):
  data_object =  NumpyToPytorchDataset(f_branch__np, k_branch__np, h_branch__np, trunk__np, output__np, device, batch_size)
  ds = DataLoader(data_object, batch_size=batch_size, shuffle=if_shuffle)
  return ds


# ## Define models

# In[5]:


"""
K-branch (FCN)
"""
class K_BranchNetCNN(nn.Module):
    def __init__(self, d_in, dim):
        super(K_BranchNetCNN, self).__init__()
        self.d_in = d_in
        self.dim = dim

        
        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(1, 2 * dim, (4, 4), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, 4 * dim, (4, 4), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, (2, 2), (2, 2), (0, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, 1, (2, 2), (2, 2), (0, 0)),
            nn.ReLU()
        )

    def forward(self, input_data):
        output = self.features_to_image(input_data)
        return output
 
#k_branchnet_dummy = K_BranchNetCNN(10, 4)
#summary(k_branchnet_dummy.to(device), input_size=(1, 10, 10))


# In[6]:


"""
F-branch (linear)
"""
class F_BranchNetLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super(F_BranchNetLinear, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.linear1 = nn.Linear(self.d_in, self.d_out)


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.linear1(x)
        return x

#f_branchnet_dummy = F_BranchNetLinear(144, 72)
#summary(f_branchnet_dummy.to(device), input_size=(1, 100), batch_size=-1)


class H_BranchNetLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super(H_BranchNetLinear, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.linear1 = nn.Linear(self.d_in, self.d_out)


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.linear1(x)

        return x

#h_branchnet_dummy = H_BranchNetLinear(144, 72)

# In[7]:


"""
Trunk net (DNN)
You might wanna replace this network with your proposed RBF network
"""
class TrunkRBFN(nn.Module):
    
    def __init__(self, n_input, n_hidden, n_output, basis_func):
        super(TrunkRBFN, self).__init__()
        self.hidden = rbf.RBF(n_input, n_hidden, basis_func)
        #self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        #x = self.predict(x)
        return x

#trunknet_dummy = TrunkRBFN(2, 72, 72, rbf.gaussian)
#trunknet_dummy = RBFLayer(in_features_dim=2,
#               num_kernels=64,
#               out_features_dim=64,
#               radial_function=rbf_gaussian,
#               norm_function=l_norm,
#               normalization=True,
#               constant_weights_parameters=True,
#               initial_weights_parameters= torch.ones(64))
"""
wandb hyperparameter dictionary
"""
sweep_config = {"method": "grid"}
metric = {"name": "valset_l2_error",
          "goal": "minimize"}
sweep_config["metric"] = metric

max_epoch = 5000    # use at least 500 epochs
parameters_dict = {
    "trunk_in": {"values": [2]},
    "trunk_h": {"values": [72]},
    "trunk_out": {"values": [72]},
    "k_branch_ch_multiplier": {"values": [8]},
    "f_branch_output_dim": {"values": [72]},
    "activation": {"values": ["relu"]},
    "max_epoch": {"values": [max_epoch]},
    "log_freq": {"values": [20]},
    "wandb_img_freq": {"values": [20]},
    "training_batch_size": {"values": [n_temp_sensor*int(n_train/300)]},
    "val_batch_size": {"values": [1024*int(n_val/100)]}
    }
sweep_config["parameters"] = parameters_dict

import pprint
pprint.pprint(sweep_config)

project_name = "DarcyFlow_KxFxHtoU"     # name of the wandb project
group_name = "Case1_MatrixVector_product_HNONzero_KandFTensorGrid"       # group name of the experiment for wandb
model_dir = f"./models/{group_name}_KBranchFCN_MiniBatch"             # directory name of the Google Drive folder
import os
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

sweep_id = wandb.sweep(sweep_config, project=f"{project_name}")

"""
helpfer function for plotting
"""
def plot_subplots(n_samples, f_fullfield, k_fullfield, h_fullfield, u_fullfield, output_predictions, t, random_sample_idx, if_worst, worst_idx, if_best, best_idx):
  n_rows = n_samples
  n_cols = 6
  img_size = 32
  NN = img_size**2
  fig, axs = plt.subplots(n_rows, n_cols, figsize=(20,20))
  for ii in range(n_rows):
    if if_worst==True:
      pp = worst_idx[ii]
    elif if_best==True:
      pp = best_idx[ii]
    else:
      pp = random_sample_idx[ii]

    for jj in range(n_cols):
      # source
      if jj==0:
        im = axs[ii,jj].imshow(f_fullfield[n_val_start+pp,:,:], cmap="inferno")
        fig.colorbar(im, ax=axs[ii,jj])
      # kappa
      if jj==1:
        im = axs[ii,jj].imshow(k_fullfield[n_val_start+pp,:,:], cmap="inferno")
        fig.colorbar(im, ax=axs[ii,jj])
      # flux bc
      if jj==2:
        im = axs[ii, jj].plot(h_fullfield[pp, 0, :], color="red", label="top")
        im = axs[ii, jj].plot(h_fullfield[pp, -1, :], color="blue", label="bottom")
        #im = axs[ii, jj].plot(h_fullfield_1d[pp, :Nx], color="red", label="top")
        #im = axs[ii, jj].plot(h_fullfield_1d[pp, Nx:], color="blue", label="bottom")
        axs[ii, jj].legend()
      # true pressure
      if jj==3:
        im = axs[ii,jj].imshow(u_fullfield[n_val_start+pp,:,:], cmap="inferno")
        fig.colorbar(im, ax=axs[ii,jj])
        #im=axs[ii,jj].imshow(output_val_np[pp*NN:(pp+1)*NN].reshape([img_size, img_size]), cmap="inferno")
        #fig.colorbar(im, ax=axs[ii,jj])
      # predicted pressure
      if jj==4:
        im = axs[ii,jj].imshow(output_predictions[pp, :, :], vmin=u_fullfield[n_val_start+pp, :, :].min(), vmax=u_fullfield[n_val_start+pp, :, :].max(), cmap="inferno")
        #im=axs[ii,jj].imshow(output_pred_val_np[pp*NN:(pp+1)*NN].reshape([img_size, img_size]), 
        #                      vmin=output_val_np[pp*NN:(pp+1)*NN].min(), vmax=output_val_np[pp*NN:(pp+1)*NN].max(), cmap="inferno")
        fig.colorbar(im, ax=axs[ii,jj])
      # error
      if jj==5:
        error_ = u_fullfield[n_val_start+pp, :, :] - output_predictions[pp, :, :]
        im=axs[ii,jj].imshow(error_, cmap="inferno")
        fig.colorbar(im, ax=axs[ii,jj])
        axs[ii,jj].set_title(f"{100*(np.linalg.norm(error_)/np.linalg.norm(u_fullfield[n_val_start+pp, :, :])):.2f}")

  if if_worst==True:
    wandb.log({"during_training/worst_samples": fig, "epoch": t})
  elif if_best==True:
    wandb.log({"during_training/best_samples": fig, "epoch": t})
  else:
    wandb.log({"during_training/random_samples": fig, "epoch": t})

def trainer_minibatch(f_branch_net, k_branch_net, h_branch_net, trunk_net, criterion, optimizer, lr_scheduler,
                          train_dataloader, val_dataloader, 
                          val_batch_size, f_fullfield, k_fullfield, h_fullfield, u_fullfield,
                          max_epoch, log_freq, wandb_img_freq, activation, config, model_path):
    """
    function for minibatch training
    f_branch_net: branch net object for source 
    k_branch_net: branch net object for material properties
    h_branch_net: branch net object for source 
    trunk_net: trunk net object
    criterion: training criteria for loss function
    optimizer: type of optimizer to be used (Adam, RMSprop, SGD, etc.)
    lr_scheduler: scheduler to reduce step size (turned off for now)
    train_dataloader: pytorch dataloader for training data
    val_dataloader: pytorch dataloader for validation data
    val_batch_size: batch size for validation data
    f_fullfield: numpy array of fullfield source data for plotting
    k_fullfield: numpy array of fullfield material property data for plotting
    h_fullfield: numpy array of fullfield source data for plotting
    u_fullfield: numpy array of fullfield temp. data
    max_epoch: number of epochs
    log_freq: frequency of logging results
    wandb_img_freq: frequency for logging images to wandb
    activation: type of activation (ReLU/sine/tanh/alternate SineReLU)
    config: config dictionary for wandb
    model_path: path to store the checkpoints

    activation = possible options: sin/ReLU/tanh/alternate_sinerelu 
    """
    loss_np = np.zeros((int(max_epoch/log_freq), 2))
    wandb.watch((f_branch_net, k_branch_net, h_branch_net, trunk_net), criterion=criterion, log_freq=wandb_img_freq*2)
    min_val = 99999
    n_worst, n_random, n_best = 10, 10, 10
    print('TEST')
    random_sample_idx = np.random.choice(n_val, n_random, replace=False)

    for t in range(max_epoch):
        for idx, (f_train, k_train, h_train, t_train, o_train) in enumerate(train_dataloader):
          print(f"Epoch/Minibatch id = {t}/{idx}")

          # branch part
          f_out = f_branch_net(f_train)
          f_out_unsqueezed = torch.unsqueeze(f_out, dim=-1)
          h_out = h_branch_net(h_train)
          h_out_unsqueezed = torch.unsqueeze(h_out, dim=-1)
          k_out = k_branch_net(k_train).squeeze()
          kf_branch_out = torch.matmul(k_out, f_out_unsqueezed).squeeze()
          kh_branch_out = torch.matmul(k_out, h_out_unsqueezed).squeeze()
          branch_out = kf_branch_out + kh_branch_out

          # trunk part
          trunk_out = trunk_net(t_train)    
          # prediction and loss
          output_pred = torch.sum((branch_out*trunk_out), axis=1)
          train_loss = criterion(output_pred, o_train)          
          # Zero out gradients, perform a backward pass, and update the weights.
          optimizer.zero_grad()
          train_loss.backward()
          optimizer.step()
          #lr_scheduler.step()        
            
        # logging stuff
        if (t==0 or (((t+1) % log_freq) == 0)):
            loss_np[t//log_freq, 0] = t
            loss_np[t//log_freq, 1] = train_loss
            wandb.log({"train loss": train_loss, "epoch": t})            
        print(f"Epoch={t} | Training loss={train_loss:.5f}")        

        # validation
        if (t==0 or (((t+1) % wandb_img_freq) == 0)):
            val_examples_loss_list = []
            output_predictions = np.zeros([n_val, Ny, Nx])
            for val_idx, (f_val, k_val, h_val,  t_val, o_val) in enumerate(val_dataloader):
                kf_branch_out_val = torch.matmul(torch.squeeze(k_branch_net(k_val)), torch.unsqueeze(f_branch_net(f_val), dim=-1)).squeeze()
                kh_branch_out_val = torch.matmul(torch.squeeze(k_branch_net(k_val)), torch.unsqueeze(h_branch_net(h_val), dim=-1)).squeeze()
                branch_out_val = kf_branch_out_val + kh_branch_out_val
                # trunk part
                trunk_out_val = trunk_net(t_val)
                # prediction and loss
                output_pred_val = torch.sum((branch_out_val*trunk_out_val), axis=1)       # predicted
                output_pred_val_np = output_pred_val.cpu().detach().numpy().squeeze()     
                output_val_np = o_val.cpu().detach().numpy()        # true
                error_output_val = output_val_np - output_pred_val_np
                output_predictions[val_idx*int(val_batch_size/1024):(val_idx+1)*int(val_batch_size/1024), :, :] = np.reshape(output_pred_val_np, [int(val_batch_size/1024), Ny, Nx])
                for ii_val in range(int(val_batch_size/(32*32))):
                  val_examples_loss_list.append(100*(np.linalg.norm(error_output_val[32*32*ii_val:(ii_val+1)*32*32])/np.linalg.norm(output_val_np[32*32*ii_val:(ii_val+1)*32*32])))

            val_example_loss = np.array(val_examples_loss_list)
            val_l2error = np.mean(np.abs(val_example_loss))
            if val_l2error < min_val:
                min_val = val_l2error
                torch.save({
                'f_branch_net_state_dict': f_branch_net.state_dict(), 
                'k_branch_net_state_dict': k_branch_net.state_dict(), 
                'h_branch_net_state_dict': h_branch_net.state_dict(), 
                'trunk_net_state_dict': trunk_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f"{model_path}/BestModel{t}.pth")
            wandb.log({"val loss": val_l2error, "epoch": t})

            fig0, axs0 = plt.subplots()
            axs0.hist(val_example_loss)
            axs0.set_xlabel("val error (in %)")
            wandb.log({"during_training/val_histogram": fig0, "epoch": t})

            # 10-larget val. error samples
            worst_idx = np.argsort(val_example_loss)[-n_worst:]
            # 10-smallest val. error samples
            best_idx = np.argsort(val_example_loss)[:n_best]
            # worst 10 samples from the test set
            plot_subplots(n_worst, f_fullfield, k_fullfield, h_fullfield, u_fullfield, output_predictions, t, random_sample_idx, if_worst=True, worst_idx=worst_idx, if_best=False, best_idx=0)    
            # random 10 samples from the test set
            plot_subplots(n_random, f_fullfield, k_fullfield, h_fullfield, u_fullfield, output_predictions, t, random_sample_idx, if_worst=False, worst_idx=0, if_best=False, best_idx=0)         
            # best 10 samples from the test set
            plot_subplots(n_best, f_fullfield, k_fullfield, h_fullfield, u_fullfield, output_predictions, t, random_sample_idx, if_worst=False, worst_idx=0, if_best=True, best_idx=best_idx)    


    return min_val

"""
This cell initializes wandb config file, optimizer, scheduler, data loaders, and stores trained models
"""
import time
t1 = time.time()
# helper function
def count_parameters(model):
    if type(model)==list:
      #print("The provided object is a list")
      count = 0
      for i in range(len(model)):
        count += sum(p.numel() for p in model[i] if p.requires_grad)
    return count

def run_trainer(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, group=f"{group_name}", 
                    tags=["fGP_kGP", "H_NONzero", "case1", "matrix_vector_product", f"train{n_train}_test{n_val}", "fixedLR", "KBranchFCN", "MiniBatchTraining", "KndFTensorGrid"]):
        config = wandb.config

        # hyper-parameters for network architecture and training
        k_branch_in, k_branch_ch_multiplier  = 12, config.k_branch_ch_multiplier
        f_branch_in, f_branch_out = 144, config.f_branch_output_dim
        trunk_in, trunk_h, trunk_out  = config.trunk_in, config.trunk_h, config.trunk_out         
        max_epoch = config.max_epoch
        log_freq = config.log_freq
        wandb_img_freq = config.wandb_img_freq
        activation = "rbf"
        training_batch_size, val_batch_size = config.training_batch_size, config.val_batch_size
        # model path to the saved model
        model_path = f"{model_dir}/chkpt_FBranchOut{f_branch_out}_KBranchChMultiplier{k_branch_ch_multiplier}_KbranchOutDim74x74_TrunkOut{trunk_out}_ACT{config.activation}_Epoch{max_epoch}"
        if not os.path.exists(model_path):
          os.makedirs(model_path)

        # branch and trunk Networks
        f_branch_net = F_BranchNetLinear(f_branch_in, f_branch_out).to(device)
        h_branch_net = H_BranchNetLinear(f_branch_in, f_branch_out).to(device)
        k_branch_net = K_BranchNetCNN(k_branch_in, k_branch_ch_multiplier).to(device)
        trunk_net = TrunkRBFN(trunk_in, trunk_h, trunk_out, rbf.gaussian).to(device)
        #trunk_net = RBFLayer(in_features_dim=trunk_in,
        #               num_kernels=trunk_h,
        #               out_features_dim=trunk_out,
        #               radial_function=rbf_gaussian,
        #               norm_function=l_norm,
        #               normalization=True,
        #               constant_weights_parameters=True,
        #               initial_weights_parameters= torch.ones(64)).to(device)

        # optimization 
        criterion = torch.nn.MSELoss(reduction='mean')
        params_to_optimize = list(f_branch_net.parameters()) + list(k_branch_net.parameters()) + list(h_branch_net.parameters()) + list(trunk_net.parameters())
        params_to_optimize_f = list(f_branch_net.parameters()) 
        params_to_optimize_k =  list(k_branch_net.parameters())
        params_to_optimize_h = list(h_branch_net.parameters()) 
        params_to_optimize_trunk = list(trunk_net.parameters())
        print(f"No. of trainable params = {count_parameters(params_to_optimize)}")
        print(f"No. of f params = {count_parameters(params_to_optimize_f)}")
        print(f"No. of k params = {count_parameters(params_to_optimize_k)}")
        print(f"No. of h params = {count_parameters(params_to_optimize_h)}")
        print(f"No. of trunk params = {count_parameters(params_to_optimize_trunk)}")
        optimizer = torch.optim.Adam(params_to_optimize) 
        #optimizer = torch.optim.AdamW(params_to_optimize) 
        decayRate = 0.99
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        # data loaders
        train_dataloader = create_dataloader(f_branch_train_np, k_branch_train_np, h_branch_train_np, trunk_train_np, output_train_np, device, training_batch_size, if_shuffle=False)
        val_dataloader = create_dataloader(f_branch_val_np, k_branch_val_np, h_branch_val_np, trunk_val_np, output_val_np, device, val_batch_size, if_shuffle=False)

        # mini batch training and logging
        min_val = trainer_minibatch(f_branch_net, k_branch_net, h_branch_net, trunk_net, criterion, optimizer, lr_scheduler,
                          train_dataloader, val_dataloader, val_batch_size, f_fullfield, k_fullfield, h_fullfield, u_fullfield,
                          max_epoch, log_freq, wandb_img_freq, activation, config, model_path)

        # save last checkpoint
        torch.save({
                'f_branch_net_state_dict': f_branch_net.state_dict(), 
                'k_branch_net_state_dict': k_branch_net.state_dict(), 
                'h_branch_net_state_dict': h_branch_net.state_dict(), 
                'trunk_net_state_dict': trunk_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },  f"{model_path}/End.pth")


        print(f"total elapsed time = {time.time() - t1}")
        l2_error_s = min_val   
        print(f"Min valset error in l2 sense (in %) = {min_val}")
        wandb.run.summary["valset_l2_error"] = l2_error_s
        wandb.run.summary["no_of_params"] = count_parameters(params_to_optimize)


"""
This cell starts training
"""
wandb.agent(sweep_id, run_trainer)

