import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

    X = np.zeros([NN*NN*n_samples, 3])
    xv, yv, zv = np.meshgrid(np.linspace(0, 1, NN), np.linspace(0, 1, NN), np.linspace(0, 1, n_samples))
    z_idx = np.broadcast_to(np.arange(n_samples), (NN, n_samples))
    x_idx = np.transpose(np.broadcast_to(np.arange(NN), (n_samples, NN)))
    left_idx = np.ravel_multi_index((x_idx.flatten(), [0]*(NN*n_samples), z_idx.flatten() ), (NN,NN,n_samples))
    remain_idx = np.delete(np.arange(NN*NN*n_samples), left_idx)
    X[:,0] = xv.flatten()
    X[:,1] = yv.flatten()
    X[:,2] = zv.flatten()

    K = RBF(X, length_scale, output_scale)

    # Get components of covariance
    gamma_12 = K[remain_idx, :][:, left_idx]
    gamma_11 = K[remain_idx, :][:, remain_idx]
    gamma_22 = K[left_idx, :][:, left_idx]
    gamma_22_inv = np.linalg.inv(gamma_22)
    # Schur complement of gamma_22
    b_11_inv = gamma_11 - gamma_12 @ gamma_22_inv @ np.transpose(gamma_12)
    # print(b_11_inv)

    Q = np.linalg.cholesky(b_11_inv + 1e-3*np.eye(remain_idx.size))
    # L = np.linalg.cholesky(K + 1e-6*np.eye(NN*NN*n_samples))
    gp_samples = (Q @ np.random.randn(remain_idx.size, 1)) + gp_min

    output = np.zeros((NN*NN*n_samples,))
    output[remain_idx] = gp_samples.flatten()
    # min_v, max_v = gp_samples.min(), gp_samples.max()

    normalized_sm = output.reshape((NN,NN,n_samples))
    # normalized_sm = (gp_samples - min_v)/(max_v - min_v) + gp_min * (1 - (gp_samples - min_v)/(max_v - min_v))
    return normalized_sm

DIM = 17
N_SAMPLES = 1

output = normalized_gp_samples(
    DIM,
    N_SAMPLES,
    0.2,
    0.5,
    0.
).reshape((DIM, DIM, N_SAMPLES))

fig, ax = plt.subplots()

im = ax.pcolor(output[:,:,0], cmap=plt.cm.seismic, vmin=-3, vmax=3)
# fig.colorbar()


def update(frame):
    scooby = im.set_array(output[:,:,frame])
    return scooby


ani = animation.FuncAnimation(fig=fig, func=update, frames=N_SAMPLES, interval=100)
plt.show()
