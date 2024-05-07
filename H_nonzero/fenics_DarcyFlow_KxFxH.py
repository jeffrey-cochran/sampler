#!/usr/bin/env python
# coding: utf-8

#try:
import dolfin  7c1ccccchvb
#except ImportError:
#    get_ipython().system('wget "https://fem-on-colab.github.io/releases/fenics-install.sh" -O "/tmp/fenics-install.sh" && bash "/tmp/fenics-install.sh"')
#    import dolfin


# In[ ]:


# Import libraries
import os
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm.notebook import tqdm
from torchsummary import summary
import dolfin as dl
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

# Ensure deterministic behavior
seed_no = 1008
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % seed_no)
np.random.seed(hash("improves reproducibility") % seed_no)
torch.manual_seed(hash("by removing stochasticity") % seed_no)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % seed_no)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ## Generate GP random fields for k, f, and h

# In[ ]:


# GP random field for f and k
def normalized_gp_samples(NN, n_samples, length_scale, output_scale, gp_min):
    # GRF sample generation
    def RBF(x, length_scale, output_scale):
        N = x.shape[0]
        dist_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dist_mat[i,j] = np.linalg.norm(x[i,:] - x[j,:])
        r = (dist_mat/length_scale)**2        
        return output_scale * np.exp(-0.5 * r)

    X = np.zeros([NN**2, 2])
    xv, yv = np.meshgrid(np.linspace(0, 1, NN), np.linspace(0, 1, NN))
    X[:,0] = xv.flatten()
    X[:,1] = yv.flatten()

    K = RBF(X, length_scale, output_scale)
    L = np.linalg.cholesky(K + 1e-10*np.eye(NN**2))
    gp_samples = (L @ np.random.randn(NN**2, n_samples)) + gp_min
    min_v, max_v = gp_samples.min(), gp_samples.max()

    #normalized_sm = gp_samples
    normalized_sm = (gp_samples - min_v)/(max_v - min_v) + gp_min * (1 - (gp_samples - min_v)/(max_v - min_v))
    return normalized_sm




def normalized_gp_samples1D(NN, n_samples, length_scale, output_scale):
    # GRF sample generation
    def RBF1d(x1, x2, length_scale, output_scale):
        diffs = np.expand_dims(x1 / length_scale, 1) - \
                np.expand_dims(x2 / length_scale, 0)
        r2 = np.sum(diffs**2, axis=2)
        return output_scale * np.exp(-0.5 * r2)

    X = np.linspace(0, 1, NN)[:, None]
    K = RBF1d(X, X, length_scale, output_scale)
    L = np.linalg.cholesky(K + 1e-myg7 c*np.eye(NN))
    gp_samples = L @ np.random.randn(NN, n_samples) 
    min_v, max_v = gp_samples.min(), gp_samples.max()
    normalized_gp = (gp_samples - min_v)/(max_v - min_v)  # normalize between min_v and max_v

    return (2*normalized_gp) - 1.     # normalize between -1 and 1



NN  = 32
n_samples = 30 #10000 #30#10 #CHANGED
gp_min = 0.02
f_length_scale = 0.2
f_output_scale = 2000
k_length_scale = 0.4
k_output_scale = 1000
h_length_scale = 0.3
h_output_scale = 1.0

# f_gp_samples
f_gp_samples = normalized_gp_samples(NN, n_samples, f_length_scale, f_output_scale, gp_min)
f_gp_samples_3d = f_gp_samples.T.reshape([n_samples, NN, NN])
# k_gp_samples
k_gp_samples = normalized_gp_samples(NN, n_samples, k_length_scale, k_output_scale, gp_min)
k_gp_samples_3d = k_gp_samples.T.reshape([n_samples, NN, NN])
# h_gp_samples
h_gp_1d = normalized_gp_samples1D(NN, 2*n_samples, h_length_scale, h_output_scale)
h_gp_samples_3d = np.zeros([n_samples, NN, NN])
for n in range(n_samples):
    h_gp_samples_3d[n, 0, :] = h_gp_1d[:, 2*n]
    h_gp_samples_3d[n, -1, :] = h_gp_1d[:, (2*n)+1]



print(k_gp_samples_3d.shape, k_gp_samples_3d.max(), k_gp_samples_3d.min())
print(f_gp_samples_3d.shape, f_gp_samples_3d.max(), f_gp_samples_3d.min())
print(h_gp_1d.shape, h_gp_1d.max(), h_gp_1d.min())


#for n in range(5):
#  plt.figure(figsize=(17, 5))
#  plt.subplot(141)
#  plt.imshow(f_gp_samples_3d[n, :, :])
#  plt.colorbar()
#  plt.subplot(142)
#  plt.imshow(k_gp_samples_3d[n, :, :])
#  plt.colorbar()
#  plt.subplot(143)
#  plt.imshow(h_gp_samples_3d[n, :, :])
#  plt.colorbar()
#  plt.subplot(144)
#  plt.plot(h_gp_1d[:, 2*n])
#  plt.plot(h_gp_1d[:, (2*n)+1])


# ## Forward problem solver

# In[ ]:


def solver_nonzero_neumann_numpy(kappa, source, neumann_bc, Nx, Ny, degree=1):
    """
    Solve -Laplace(u) = f on [0,1] x [0,1] with 2*Nx*Ny Lagrange
    elements of specified degree and u = u_D (Expresssion) on
    the boundary.
    """
    # Create mesh and define function space
    mesh = dl.UnitSquareMesh(Nx, Ny)
    Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vu = dl.FunctionSpace(mesh, 'Lagrange', 1)
    
    # define function for state and adjoint
    u = dl.Function(Vu)
    m = dl.Function(Vm)
    f = dl.Function(Vm)
    g = dl.Function(Vm)
    
    # define Trial and Test Functions
    u_trial, m_trial  = dl.TrialFunction(Vu), dl.TrialFunction(Vm)
    u_test,  m_test    = dl.TestFunction(Vu),  dl.TestFunction(Vm)
    
    # Define boundary condition
    def boundary_d(x,on_boundary):
        return (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS) and on_boundary
        #return (x[0] < dl.DOLFIN_EPS) and on_boundary
    bc_state = [dl.DirichletBC(Vu, dl.Constant(0.), boundary_d)]


    # Interpolate values
    k_mat = dl.Constant(1)
    k_mat = dl.interpolate(k_mat, Vm)    
    v2d_k = dl.vertex_to_dof_map(Vm)
    k_mat.vector()[v2d_k[:]] = (kappa).flatten()  

    f_mat = dl.Constant(1)
    f_mat = dl.interpolate(f_mat, Vm)    
    v2d_f = dl.vertex_to_dof_map(Vm)
    f_mat.vector()[v2d_f[:]] = (source).flatten() 

    g_mat = dl.Constant(1)
    g_mat = dl.interpolate(g_mat, Vm)    
    v2d_g = dl.vertex_to_dof_map(Vm)
    g_mat.vector()[v2d_g[:]] = (neumann_bc).flatten()    

    
    ## Variational forms 
    # forward problem
    a_state = dl.inner( m * dl.grad(u_trial), dl.grad(u_test)) * dl.dx
    l_state = (f * u_test * dl.dx) + (g * u_test * dl.ds)
   

    ## Solving system of equations
    # forward problem
    m.assign(k_mat)
    f.assign(f_mat)
    g.assign(g_mat)
    A_state, L_state = dl.assemble_system(a_state, l_state, bc_state)
    dl.solve(A_state, u.vector(), L_state)
    
    # reshape 
    mesh1 = u.function_space().mesh()
    C = np.reshape(u.compute_vertex_values(mesh1), (Ny+1, Nx+1))   

    return C



# In[ ]:


k_samples = 25#4 CHANGED
f_samples = 25#3 CHANGED
h_samples = 25#2 CHANGED
n_paired_samples = k_samples * f_samples * h_samples
Ny = Nx = NN
t1 = time.time()

k_fullfield = np.zeros([n_paired_samples, Ny, Nx])
f_fullfield = np.zeros([n_paired_samples, Ny, Nx]) 
h_fullfield = np.zeros([n_paired_samples, Ny, Nx])
u_fullfield = np.zeros([n_paired_samples, Ny, Nx])



for kk in range(h_samples): # bc loop
  for ii in range(k_samples): # kappa loop
    for jj in range(f_samples):   # source loop
      u_fem_np = solver_nonzero_neumann_numpy(k_gp_samples_3d[ii, :, :], f_gp_samples_3d[jj, :, :], h_gp_samples_3d[kk, :, :], Nx-1, Ny-1, degree=1)

      
      k_fullfield[(kk*f_samples*k_samples)+(ii*f_samples)+jj, :, :] = k_gp_samples_3d[ii, :, :]
      f_fullfield[(kk*f_samples*k_samples)+(ii*f_samples)+jj, :, :] = f_gp_samples_3d[jj, :, :] 
      h_fullfield[(kk*f_samples*k_samples)+(ii*f_samples)+jj, :, :] = h_gp_samples_3d[kk, :, :]    
      u_fullfield[(kk*f_samples*k_samples)+(ii*f_samples)+jj, :, :] = u_fem_np
  
print(time.time()-t1)

#Ny = Nx = NN
#t1 = time.time()

#k_fullfield = np.copy(k_gp_samples_3d)  # These are now the same
#f_fullfield = np.copy(f_gp_samples_3d)  # These are now the same
#h_fullfield = np.copy(h_gp_samples_3d)  # These are now the same
#u_fullfield = np.zeros([n_samples, Ny, Nx])
#
#
#
#for i in range(n_samples): # single loop
#  u_fullfield[i,:,:] = solver_nonzero_neumann_numpy(k_gp_samples_3d[i, :, :], f_gp_samples_3d[i, :, :], h_gp_samples_3d[i, :, :], Nx-1, Ny-1, degree=1)    
  
#print(time.time()-t1)

#fig, axs = plt.subplots(n_paired_samples, 4, figsize=(16, 80))
#for ii in range(n_paired_samples):
#  im = axs[ii, 0].imshow(f_fullfield[ii, :, :])
#  fig.colorbar(im, ax=axs[ii, 0])
#  im = axs[ii, 1].imshow(k_fullfield[ii, :, :])
#  fig.colorbar(im, ax=axs[ii, 1])
#  im = axs[ii, 2].imshow(u_fullfield[ii, :, :])
#  fig.colorbar(im, ax=axs[ii, 2])
#  im = axs[ii, 3].plot(h_fullfield[ii, 0, :], color="red", label="top")
#  im = axs[ii, 3].plot(h_fullfield[ii, -1, :], color="blue", label="bottom")
#  axs[ii, 3].legend()
  #im = axs[ii, 4].imshow(u_fullfield_train[ii, :, :])
  #fig.colorbar(im, ax=axs[ii, 4])


# In[ ]:


save_main_dir = "./fullfield_data"
if not os.path.exists(save_main_dir):
  os.makedirs(save_main_dir)


# In[ ]:


np.save(f"{save_main_dir}/f_fullfield.npy", f_fullfield)
np.save(f"{save_main_dir}/k_fullfield.npy", k_fullfield)
np.save(f"{save_main_dir}/h_fullfield.npy", h_fullfield)
np.save(f"{save_main_dir}/u_fullfield.npy", u_fullfield)
print(f_fullfield.shape, f_fullfield.min(), f_fullfield.max())
print(k_fullfield.shape, k_fullfield.min(), k_fullfield.max())
print(h_fullfield.shape, h_fullfield.min(), h_fullfield.max())
print(u_fullfield.shape, u_fullfield.min(), u_fullfield.max())


# ## Sparse data | Generating branch, trunk, output data at sensor locations

# In[ ]:


x_vec_ = np.append(np.arange(0, 32, 3), 31) 
y_vec_ = np.append(np.arange(0, 32, 3), 31) 
print(x_vec_)

import numpy 
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

output_idx = cartesian_product(x_vec_, y_vec_)
f_idx = output_idx
k_idx = output_idx
h_idx = output_idx
output_idx

# In[ ]:


n_samples = u_fullfield.shape[0]
n_temp_int_sensor = 144
n_bnd_sensor = (2*32) + (2*30)
n_temp_sensor = n_bnd_sensor + n_temp_int_sensor


f_at_sensor = f_fullfield[:, f_idx[:,0], f_idx[:,1]]
k_at_sensor = k_fullfield[:, k_idx[:,0], k_idx[:,1]]
h_at_sensor = h_fullfield[:, h_idx[:,0], h_idx[:,1]]
f_branch_data = np.repeat(f_at_sensor, repeats=n_temp_sensor, axis=0)
k_branch_data = np.repeat(k_at_sensor, repeats=n_temp_sensor, axis=0)
h_branch_data = np.repeat(h_at_sensor, repeats=n_temp_sensor, axis=0)

#plt.imshow(f_fullfield[0,:,:], origin="lower")
#plt.plot(f_idx[:, 1], f_idx[:, 0], "r x")
#plt.colorbar()
print(f_idx)
print("--")
print(f_at_sensor[0,:])


# In[ ]:


#plt.imshow(k_fullfield[0,:,:], origin="lower")
#plt.plot(k_idx[:, 1], k_idx[:, 0], "r x")
#plt.colorbar()
print(k_idx)
print("--")
print(k_at_sensor[0,:])


# In[ ]:


n_f_sensor = 144
n_k_sensor = 144
n_h_sensor = 144
n_temp_sensor = 268
sparse_data_dir = f"./sparse_data/sensorgrid12x12_Nf{n_f_sensor}_Nk{n_k_sensor}_Nh{n_h_sensor}_Ntemp{n_temp_sensor}"
#sparse_sensorloc_dir = f"./sparse_data/sensor_location_Nf{n_f_sensor}_Nk{n_k_sensor}_Ntemp{n_temp_sensor}" #ADDED
if not os.path.exists(sparse_data_dir):
  os.makedirs(f"{sparse_data_dir}")
#if not os.path.exists(sparse_sensorloc_dir): #ADDED
#  os.makedirs(f"{sparse_sensorloc_dir}")


# In[ ]:


# BRANCH, TRUNK, and OUTPUT Data
n_samples = u_fullfield.shape[0]
n_temp_int_sensor = 144
n_bnd_sensor = (2*32) + (2*30)
n_temp_sensor = n_bnd_sensor + n_temp_int_sensor

#n_f_sensor = 144
#n_k_sensor = 144
#h_idx = np.linspace(1, 2*NN-1, n_h_sensor).astype(int)
#f_idx = np.random.randint(low=0, high=f_fullfield.shape[2], size=[n_f_sensor, 2])
#k_idx = np.random.randint(low=0, high=k_fullfield.shape[2], size=[n_k_sensor, 2])
#h_fullfield_1d = np.zeros([n_paired_samples, 2*NN])
#for n in range(n_paired_samples):
#  h_fullfield_1d[n,:NN] = h_fullfield[n,0,:]  
#  h_fullfield_1d[n,NN:] = h_fullfield[n,-1,:]  

f_at_sensor = f_fullfield[:, f_idx[:,0], f_idx[:,1]]
k_at_sensor = k_fullfield[:, k_idx[:,0], k_idx[:,1]]
h_at_sensor = h_fullfield[:, h_idx[:,0], h_idx[:,1]]
#h_at_sensor = h_fullfield_1d[:, h_idx]
f_branch_data = np.repeat(f_at_sensor, repeats=n_temp_sensor, axis=0)
k_branch_data = np.repeat(k_at_sensor, repeats=n_temp_sensor, axis=0)
h_branch_data = np.repeat(h_at_sensor, repeats=n_temp_sensor, axis=0)
np.save(f"./sparse_data/sensorgrid12x12_Nf{n_f_sensor}_Nk{n_k_sensor}_Nh{n_h_sensor}_Ntemp{n_temp_sensor}/f_idx.npy", f_idx)
np.save(f"./sparse_data/sensorgrid12x12_Nf{n_f_sensor}_Nk{n_k_sensor}_Nh{n_h_sensor}_Ntemp{n_temp_sensor}/k_idx.npy", k_idx)
np.save(f"./sparse_data/sensorgrid12x12_Nf{n_f_sensor}_Nk{n_k_sensor}_Nh{n_h_sensor}_Ntemp{n_temp_sensor}/h_idx.npy", h_idx)
np.save(f"./sparse_data/sensorgrid12x12_Nf{n_f_sensor}_Nk{n_k_sensor}_Nh{n_h_sensor}_Ntemp{n_temp_sensor}/f_branch_data.npy", f_branch_data)
np.save(f"./sparse_data/sensorgrid12x12_Nf{n_f_sensor}_Nk{n_k_sensor}_Nh{n_h_sensor}_Ntemp{n_temp_sensor}/k_branch_data.npy", k_branch_data)
np.save(f"./sparse_data/sensorgrid12x12_Nf{n_f_sensor}_Nk{n_k_sensor}_Nh{n_h_sensor}_Ntemp{n_temp_sensor}/h_branch_data.npy", h_branch_data)
#np.save(f"./f_idx.npy", f_idx)
#np.save(f"./k_idx.npy", k_idx)
#np.save(f"./f_branch_data.npy", f_branch_data)
#np.save(f"./k_branch_data.npy", k_branch_data)




Nx = Ny = 32
xvec_domain, yvec_domain = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))
def boundary_trunk_output(n_bnd_sensor, temp_array):
    # Assume origin at bottom left corner
    # ====== Trunk array ======
    trunk_bnd = np.zeros([n_bnd_sensor, 2])
    # bottom row
    trunk_bnd[:32, 0] = xvec_domain[0,:] 
    trunk_bnd[:32, 1] = yvec_domain[0,:]
    # left row
    trunk_bnd[32:32+31, 0] = xvec_domain[1:, -1]
    trunk_bnd[32:32+31, 1] = yvec_domain[1:, -1]
    # top row
    trunk_bnd[32+31:32+31+31, 0] = xvec_domain[-1, :-1]
    trunk_bnd[32+31:32+31+31, 1] = yvec_domain[-1, :-1]
    # right row
    trunk_bnd[32+31+31:, 0] = xvec_domain[1:-1, 0]
    trunk_bnd[32+31+31:, 1] = yvec_domain[1:-1, 0]
    # ====== Output array ======
    temperature_data = temp_array
    output_bnd = np.zeros([n_bnd_sensor, 1])
    output_bnd[:32, 0] = temperature_data[0,:]
    output_bnd[32:32+31, 0] = temperature_data[1:, -1]
    output_bnd[32+31:32+31+31, 0] = temperature_data[-1, :-1]
    output_bnd[32+31+31:, 0] = temperature_data[1:-1, 0]

    return trunk_bnd, output_bnd


trunk_data = np.zeros([n_samples*n_temp_sensor, 2])
output_data = np.zeros([n_samples*n_temp_sensor, 1])
temp_int_list = []
for ii in range(n_samples):
  print(f"Sample = {ii}")
  temp_int_idx = np.random.randint(low=1, high=u_fullfield.shape[2]-1, size=[n_temp_int_sensor, 2])
  # trunk data
  trunk_data[ii*n_temp_sensor:(ii*n_temp_sensor)+n_bnd_sensor, :] = boundary_trunk_output(n_bnd_sensor, u_fullfield[ii,:,:])[0]
  trunk_data[(ii*n_temp_sensor)+n_bnd_sensor:(ii+1)*n_temp_sensor, 0] = xvec_domain[temp_int_idx[:,0], temp_int_idx[:,1]]
  trunk_data[(ii*n_temp_sensor)+n_bnd_sensor:(ii+1)*n_temp_sensor, 1] = yvec_domain[temp_int_idx[:,0], temp_int_idx[:,1]]
  # output data
  output_data[ii*n_temp_sensor:(ii*n_temp_sensor)+n_bnd_sensor, :] = boundary_trunk_output(n_bnd_sensor, u_fullfield[ii,:,:])[1]
  output_data[(ii*n_temp_sensor)+n_bnd_sensor:(ii+1)*n_temp_sensor, 0] = u_fullfield[ii, temp_int_idx[:,0], temp_int_idx[:,1]]
  temp_int_list.append(temp_int_idx)

np.save(f"./sparse_data/sensorgrid12x12_Nf{n_f_sensor}_Nk{n_k_sensor}_Nh{n_h_sensor}_Ntemp{n_temp_sensor}/trunk_data.npy", trunk_data)
np.save(f"./sparse_data/sensorgrid12x12_Nf{n_f_sensor}_Nk{n_k_sensor}_Nh{n_h_sensor}_Ntemp{n_temp_sensor}/output_data.npy", output_data)
#np.save(f"./trunk_data.npy", trunk_data)
#np.save(f"./output_data.npy", output_data)


# In[ ]:





# In[ ]:





# In[ ]:


n_f_sensor = 144  
n_k_sensor = 144
n_h_sensor = 144  
n_temp_sensor = 268
#os.makedirs(f"./sparse_data/sensor_location_Nf{n_f_sensor}_Nk{n_k_sensor}_Nh{n_h_sensor}_Ntemp{n_temp_sensor}")
get_ipython().run_line_magic('cd', './sparse_data/sensorgrid12x12_Nf144_Nk144_Nh144_Ntemp268/')
#h_idx = np.random.choice(2*Nx, n_h_sensor, replace=False)
#h_idx = np.linspace(1, 2*NN-1, n_h_sensor).astype(int)
#h_gp_reshape = h_gp_1d.T.reshape([25, 2*NN])    # 10 = no. of GP samples generated at the beginning
#np.save("./h_idx.npy", h_idx)
#np.save("./h_gp_reshape.npy", h_gp_reshape)


# In[ ]:


get_ipython().system('ls')


# In[ ]:


# BRANCH, TRUNK, and OUTPUT Data
n_samples = u_fullfield.shape[0]
n_temp_int_sensor = 25
n_bnd_sensor = (2*32) + (2*30)
n_temp_sensor = n_bnd_sensor + n_temp_int_sensor
f_idx = np.load("f_idx.npy")
k_idx = np.load("k_idx.npy")
h_idx = np.load("h_idx.npy")
#h_fullfield_1d = np.load("h_fullfield_1d.npy")

#h_fullfield_1d = np.zeros([n_paired_samples, 2*NN])
#for n in range(n_paired_samples):
#  h_fullfield_1d[n,:NN] = h_fullfield[n,0,:]  
#  h_fullfield_1d[n,NN:] = h_fullfield[n,-1,:]  

f_at_sensor = f_fullfield[:, f_idx[:,0], f_idx[:,1]]
k_at_sensor = k_fullfield[:, k_idx[:,0], k_idx[:,1]]
h_at_sensor = h_fullfield[:, h_idx[:,0], h_idx[:,1]]
#h_at_sensor = h_fullfield_1d[:, h_idx]

f_branch_data = np.repeat(f_at_sensor, repeats=n_temp_sensor, axis=0)
k_branch_data = np.repeat(k_at_sensor, repeats=n_temp_sensor, axis=0)
h_branch_data = np.repeat(h_at_sensor, repeats=n_temp_sensor, axis=0)

np.save(f"./f_branch_data.npy", f_branch_data)
np.save(f"./k_branch_data.npy", k_branch_data)
np.save(f"./h_branch_data.npy", h_branch_data)
#np.save(f"./h_fullfield_1d.npy", h_fullfield_1d)


Nx = Ny = 32
xvec_domain, yvec_domain = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))
def boundary_trunk_output(n_bnd_sensor, temp_array):
    # Assume origin at bottom left corner
    # ====== Trunk array ======
    trunk_bnd = np.zeros([n_bnd_sensor, 2])
    # bottom row
    trunk_bnd[:32, 0] = xvec_domain[0,:] 
    trunk_bnd[:32, 1] = yvec_domain[0,:]
    # left row
    trunk_bnd[32:32+31, 0] = xvec_domain[1:, -1]
    trunk_bnd[32:32+31, 1] = yvec_domain[1:, -1]
    # top row
    trunk_bnd[32+31:32+31+31, 0] = xvec_domain[-1, :-1]
    trunk_bnd[32+31:32+31+31, 1] = yvec_domain[-1, :-1]
    # right row
    trunk_bnd[32+31+31:, 0] = xvec_domain[1:-1, 0]
    trunk_bnd[32+31+31:, 1] = yvec_domain[1:-1, 0]
    # ====== Output array ======
    temperature_data = temp_array
    output_bnd = np.zeros([n_bnd_sensor, 1])
    output_bnd[:32, 0] = temperature_data[0,:]
    output_bnd[32:32+31, 0] = temperature_data[1:, -1]
    output_bnd[32+31:32+31+31, 0] = temperature_data[-1, :-1]
    output_bnd[32+31+31:, 0] = temperature_data[1:-1, 0]

    return trunk_bnd, output_bnd


trunk_data = np.zeros([n_samples*n_temp_sensor, 2])
output_data = np.zeros([n_samples*n_temp_sensor, 1])
temp_int_list = []
for ii in range(n_samples):
  print(f"Sample = {ii}")
  temp_int_idx = np.random.randint(low=1, high=u_fullfield.shape[2]-1, size=[n_temp_int_sensor, 2])
  # trunk data
  trunk_data[ii*n_temp_sensor:(ii*n_temp_sensor)+n_bnd_sensor, :] = boundary_trunk_output(n_bnd_sensor, u_fullfield[ii,:,:])[0]
  trunk_data[(ii*n_temp_sensor)+n_bnd_sensor:(ii+1)*n_temp_sensor, 0] = xvec_domain[temp_int_idx[:,0], temp_int_idx[:,1]]
  trunk_data[(ii*n_temp_sensor)+n_bnd_sensor:(ii+1)*n_temp_sensor, 1] = yvec_domain[temp_int_idx[:,0], temp_int_idx[:,1]]
  # output data
  output_data[ii*n_temp_sensor:(ii*n_temp_sensor)+n_bnd_sensor, :] = boundary_trunk_output(n_bnd_sensor, u_fullfield[ii,:,:])[1]
  output_data[(ii*n_temp_sensor)+n_bnd_sensor:(ii+1)*n_temp_sensor, 0] = u_fullfield[ii, temp_int_idx[:,0], temp_int_idx[:,1]]
  temp_int_list.append(temp_int_idx)

np.save(f"./trunk_data.npy", trunk_data)
np.save(f"./output_data.npy", output_data)


# In[ ]:


temp_int_indices = np.array(temp_int_list)
temp_int_indices.shape


# In[ ]:


from mpl_toolkits.axes_grid1 import make_axes_locatable

#fig, axs = plt.subplots(4, 4, figsize=(15, 15))
#for ii in range(4):
#  for jj in range(4):
#    im = axs[ii,jj].imshow(f_fullfield[ii*100+jj, :, :], origin="lower")
#    axs[ii,jj].axis("off")
#    divider = make_axes_locatable(axs[ii,jj])
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    fig.colorbar(im, ax=axs[ii,jj], cax=cax)
#    axs[ii,jj].plot(f_idx[:, 1], f_idx[:, 0], "r o")
#plt.suptitle("source")
#
#fig, axs = plt.subplots(4, 4, figsize=(16, 16))
#for ii in range(4):
#  for jj in range(4):
#    im = axs[ii,jj].imshow(k_fullfield[ii*100+jj, :, :].reshape([32, 32]), origin="lower")
#    axs[ii,jj].axis("off")
#    divider = make_axes_locatable(axs[ii,jj])
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    fig.colorbar(im, ax=axs[ii,jj], cax=cax)
#    axs[ii,jj].plot(k_idx[:, 1], k_idx[:, 0], "r x")
#plt.suptitle("kappa")
#
#fig, axs = plt.subplots(4, 4, figsize=(15, 15))
#for ii in range(4):
#  for jj in range(4):
#    im = axs[ii,jj].imshow(u_fullfield_train[ii*100+jj, :, :], origin="lower")
#    axs[ii,jj].axis("off")
#    divider = make_axes_locatable(axs[ii,jj])
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    fig.colorbar(im, ax=axs[ii,jj], cax=cax)
#    axs[ii,jj].plot(temp_int_indices[ii*100+jj, :, 1], temp_int_indices[ii*100+jj, :, 0], "r *")
#plt.suptitle("temp.")
#
#fig, axs = plt.subplots(4, 4, figsize=(15, 15))
#for ii in range(4):
#  for jj in range(4):
#    axs[ii, jj].plot(h_fullfield_train[ii*4+jj, :Nx], color="red", label="top")
#    axs[ii, jj].plot(h_fullfield_train[ii*4+jj, Nx:], color="blue", label="bottom")
#    axs[ii, jj].legend()
#plt.suptitle("flux")


## # ================
## # Generating and saving fullfield data for validation
## 
#
## In[ ]:
#
#
##get_ipython().run_line_magic('cd', './../../../val_data/fullfield_data')
#get_ipython().run_line_magic('cd', './../../fullfield_data')
#
#
## In[ ]:
#
#
## kf val indices
#total_idx=np.arange(10000)
#first_row = (total_idx[:100:10])
#val_idx = np.zeros([10, 10])
#for n in range(10):
#  val_idx[n, :] = first_row + (n*1000)
#kf_val_idx = val_idx.flatten().astype(np.int32)
#print(kf_val_idx, kf_val_idx.shape)
#
## h val indices
#h_total_idx = np.arange(1000)
#h_val_idx = h_total_idx[::100]
#print(h_val_idx)
#
#
#t1 = time.time()
#Ny = Nx = 32
#n_val = len(kf_val_idx)*len(h_val_idx)
##n_samples = 1000   # already defined in earlier cells
#k_fullfield_val = np.zeros([n_val, Ny, Nx])
#f_fullfield_val = np.zeros([n_val, Ny, Nx])
#u_fullfield_val = np.zeros([n_val, Ny, Nx])
#h_fullfield_val = np.zeros([n_val, 2*Nx])
#
#
#for ii in range(len(kf_val_idx)):       # KF loop for generating validation data
#    print(ii)
#    for jj in range(len(h_val_idx)):    # flux loop for generating validation data        
#        k_fullfield_val[(ii*len(h_val_idx))+jj, :, :] = k_fullfield[kf_val_idx[ii], :, :]
#        f_fullfield_val[(ii*len(h_val_idx))+jj, :, :] = f_fullfield[kf_val_idx[ii], :, :]
#        h_fullfield_val[(ii*len(h_val_idx))+jj, :] = h_fullfield[h_val_idx[jj], :]
#
#        u_fullfield_val[(ii*len(h_val_idx))+jj, :, :] = solver_nonzero_neumann_numpy(k_fullfield[kf_val_idx[ii], :, :], f_fullfield[kf_val_idx[ii], :, :], h_gp_samples_3d[h_val_idx[jj], :, :], Nx-1, Ny-1, degree=1)
#            
#
#print(time.time()-t1)
#
#
##np.save(f"./k_fullfield_val.npy", k_fullfield_val)
##np.save(f"./f_fullfield_val.npy", f_fullfield_val)
##np.save(f"./u_fullfield_val.npy", u_fullfield_val)
#
#
## In[ ]:
#
#
##for ii in range(12):
##  fig, axs = plt.subplots(len(h_val_idx), 5, figsize=(20, 20))
##  for jj in range(len(h_val_idx)):
##    im = axs[jj, 0].imshow(f_fullfield_val[(ii*len(h_val_idx))+jj, :, :])
##    fig.colorbar(im, ax=axs[jj, 0])
##    im = axs[jj, 1].imshow(k_fullfield_val[(ii*len(h_val_idx))+jj, :, :])
##    fig.colorbar(im, ax=axs[jj, 1])
##    im = axs[jj, 2].imshow(u_fullfield0[kf_val_idx[ii], :, :])
##    fig.colorbar(im, ax=axs[jj, 2])
##    #im = axs[jj, 3].plot(h_gp_1d[:, 2*h_val_idx[jj]], color="red", label="top")
##    #im = axs[jj, 3].plot(h_gp_1d[:, (2*h_val_idx[jj])+1], color="blue", label="bottom")
##    im = axs[jj, 3].plot(h_fullfield_val[(ii*len(h_val_idx))+jj, :Nx], color="red", label="top")
##    im = axs[jj, 3].plot(h_fullfield_val[(ii*len(h_val_idx))+jj, Nx:], color="blue", label="bottom")
##    axs[jj, 3].legend()
##    im = axs[jj, 4].imshow(u_fullfield_val[(ii*len(h_val_idx))+jj, :, :])
##    fig.colorbar(im, ax=axs[jj, 4])
##  plt.suptitle(f"img no = {ii}")
#
#
## In[ ]:
#
#
#get_ipython().system('pwd')
#
#
## In[ ]:
#
#
#np.save(f"./k_fullfield_val.npy", k_fullfield_val)
#np.save(f"./f_fullfield_val.npy", f_fullfield_val)
#np.save(f"./h_fullfield_val.npy", h_fullfield_val)
#np.save(f"./u_fullfield_val.npy", u_fullfield_val)
#
#
## # ===========
## # Playground from here...
#
## In[ ]:
#
#
#
#
#
## ## Verificaion with the method of manufactured solution... Test passed!
#
## In[ ]:
#
#
#Nx = Ny = 32
#xv, yv = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
#xy = np.multiply(xv, yv)
#
#k = xv + yv
#u = np.multiply(xy, np.multiply(xv-1, yv-1))
#f = (-2*yv**3) + (3*yv**2) - (4*np.multiply(xv, yv**2)) - (4*np.multiply(xv**2, yv)) + (8*np.multiply(xv, yv)) - yv - xv + (3*xv**2) - (2*xv**3)
##f = -np.multiply(6*xy, (xv**2 + yv**2 - 2))
##u = np.multiply(xy, np.multiply(xv**2 - 1, yv**2 - 1))
#
## fenics code
#kappa = k #np.ones([Ny, Nx])
#source = f
#u_fem = solver(kappa, source, 0., Nx-1, Ny-1, degree=1)
#print(u_fem.shape)
#
##plt.figure()
##plt.imshow(k, origin="lower", cmap=cm.coolwarm)
##plt.colorbar()
##plt.title("kappa = x+y")
##
##plt.figure()
##plt.imshow(f, origin="lower", cmap=cm.coolwarm)
##plt.colorbar()
##plt.title("f = -2x^3 -2y^3 + 3x^2 + 3y^2 - 4xy^2 + 8xy -4x^2y - x - y")
##
##plt.figure()
##plt.imshow(u, origin="lower", cmap=cm.coolwarm)
##plt.colorbar()
##plt.title("u = xy(x-1)(y-1)")
##
##plt.figure()
##plt.imshow(u_fem, origin="lower", cmap=cm.coolwarm)
##plt.colorbar()
##plt.title("u_fem")
##
##plt.figure()
##plt.imshow(u - u_fem, origin="lower", cmap=cm.coolwarm)
##plt.colorbar()
##plt.title("u - u_fem")


# In[ ]:




