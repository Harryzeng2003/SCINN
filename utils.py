# Import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py
import math
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt


def initial_conditions(X, x, case, config):
    x_numpy = x.detach().cpu().numpy()
    if case == 5:
        x_L, x_R, u_L, u_M, u_R = -2, 2, 2, 0, 4
        ic_ux = lambda x: np.select(
            [(x <= x_L), ((x_L < x) & (x <= x_R)), (x > x_R)], [u_L, u_M, u_R],
            default=np.nan
        )
    output_ICs = torch.tensor(ic_ux(x_numpy), requires_grad=True)
    return output_ICs


def generate_domain(N_t, N_x, N_0, tmin, tmax, xmin, xmax):
	x_values = np.linspace(xmin, xmax, N_x)
	t_values = np.linspace(tmin, tmax, N_t)
	x0_values = torch.tensor(np.linspace(xmin, xmax, N_0))
	t0_values = torch.zeros_like(x0_values)

    # Create (t, x) meshgrid
	x_grid, t_grid = np.meshgrid(x_values, t_values)

    # Flatten and convert to torch tensors
	t_final = torch.tensor(t_grid.flatten())
	x_final = torch.tensor(x_grid.flatten())
	X = torch.stack((t_final, x_final), dim=1)
	X_0 = torch.stack((t0_values, x0_values), dim=1)

	X.requires_grad = True
	X_0.requires_grad = True
	return X, X_0


def AN(a_l, a_r, h):
	W, M_1, M_2 = 17.5, 9.60, 4.22
	new = W * (a_l - a_r) - M_1 * h - M_2
	out = 1 / (1 + np.exp(-new))
	out = np.where(out >= 0.5, np.zeros_like(out), np.ones_like(out))
	return out


def result_AN(model, X, X_l, X_r, h, df):
	with torch.no_grad():  # Disable gradient since only forward pass needed
		ux_final = model(X).cpu().numpy()[:, 0:1]
		ux_left = model(X_l).cpu().numpy()[:, 0:1]
		ux_right = model(X_r).cpu().numpy()[:, 0:1]
	an_final = AN(df(ux_left), df(ux_right), h)
	return ux_final, an_final


def plot_results(X_final, U_exact, model, w_pt, case, res, df, path_images):
    with torch.no_grad():
        t, x = X_final[:, 0:1], X_final[:, 1:2]
        T_max = torch.max(t).item() if model.is_PRE else 1.0
        X_max = torch.max(x).item() if model.is_PRE else 1.0
        X_min = torch.min(x).item() if model.is_PRE else 0.0
        dx = (model.xmax - model.xmin) / (model.N_x - 1)
        t_tilde = model.v_2_vtilde(t, T_max, 0.0)
        x_tilde = model.v_2_vtilde(x, X_max, X_min)
        ux_tilde = model(torch.cat((t_tilde, x_tilde), dim=1))
        ux_final = model.vtilde_2_v(ux_tilde, model.umax, model.umin).cpu().detach().numpy()
        ux_square = ux_final.reshape((model.N_t, model.N_x, 1))
        ux_left = ux_square[:, :-2, :]
        ux_right = ux_square[:, 2:, :]
        
        temp = AN(df(ux_left), df(ux_right), dx)
        
        an_final = np.pad(temp, pad_width=((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=1)

    x_cpu = X_final.cpu().detach().numpy()[:, 1]
    t_cpu = X_final.cpu().detach().numpy()[:, 0]
 
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    plt.subplots_adjust(bottom=0.15)

    scatter_U_exact = axs[0,0].scatter(x_cpu, t_cpu, c=U_exact.cpu().detach().numpy().flatten(), s=2)
    fig.colorbar(scatter_U_exact, ax=axs[0,0])
    axs[0,0].set_xlabel('x', labelpad=0)
    axs[0,0].set_ylabel('t', labelpad=2)
    axs[0,0].set_title("(a)Exact", fontsize=12, y=-0.3) 

    scatter_ux = axs[0,1].scatter(x_cpu, t_cpu, c=ux_final.flatten(), s=2)
    fig.colorbar(scatter_ux, ax=axs[0,1])
    axs[0,1].set_xlabel('x', labelpad=0)
    axs[0,1].set_ylabel('t', labelpad=2)
    axs[0,1].set_title("(b)Predict", fontsize=12, y=-0.3)

    error = np.abs(ux_final.flatten() - U_exact.cpu().detach().numpy().flatten())
    scatter_error = axs[0,2].scatter(x_cpu, t_cpu, c=error, s=2, cmap='coolwarm')
    fig.colorbar(scatter_error, ax=axs[0,2])
    axs[0,2].set_xlabel('x', labelpad=0)
    axs[0,2].set_ylabel('t', labelpad=2)
    axs[0,2].set_title("(c)Error", fontsize=12, y=-0.3)

    res = res.cpu().detach().numpy().flatten()
    if np.isnan(res).any() or np.isinf(res).any():
        res = np.nan_to_num(res)
    scatter_pde = axs[1,0].scatter(x_cpu, t_cpu, c=res, s=2, cmap='coolwarm')
    fig.colorbar(scatter_pde, ax=axs[1,0])
    axs[1,0].set_xlabel('x', labelpad=0)
    axs[1,0].set_ylabel('t', labelpad=2)
    axs[1,0].set_title("(d)Residual", fontsize=12, y=-0.3)

    scatter_an = axs[1,1].scatter(x_cpu, t_cpu, c=an_final.flatten(), s=2)
    fig.colorbar(scatter_an, ax=axs[1,1])
    axs[1,1].set_xlabel('x', labelpad=0)
    axs[1,1].set_ylabel('t', labelpad=2)
    axs[1,1].set_title("(e)AN", fontsize=12, y=-0.3)

    scatter_w = axs[1,2].scatter(x_cpu, t_cpu, c=w_pt.cpu().detach().numpy().flatten(), s=2)
    fig.colorbar(scatter_w, ax=axs[1,2])
    axs[1,2].set_xlabel('x', labelpad=0)
    axs[1,2].set_ylabel('t', labelpad=2)
    axs[1,2].set_title("(f)RAR", fontsize=12, y=-0.3)

    # Tight layout to avoid subplot overlap
    plt.tight_layout()
 
    # Save figure
    plt.savefig(os.path.join(path_images, f"case_{case}.png"), dpi=400)
 
    # Close figure
    plt.close()


def save_results(model, path, case):
	hf = h5py.File(path + "data_case_" + str(case) + ".h5", "w")
	hf.create_dataset("loss_EQ_hist", data=np.array(model.l2_EQ_hist))
	hf.create_dataset("loss_IBC_hist", data=np.array(model.l2_IBC_hist))
	hf.create_dataset("loss_AC_hist", data=np.array(model.l2_AC_hist))
	hf.create_dataset("loss_IF_hist", data=np.array(model.l2_IF_hist))
	hf.create_dataset("loss_RH_hist", data=np.array(model.l2_RH_hist))
	hf.create_dataset("loss_all_hist", data=np.array(model.l2_all_hist))
	hf.close()


def deal_with_an(X, an_final):
    N_t, N_x, _ = an_final.shape
    X = X.reshape((N_t, N_x, 2))
    result = []
    list_k = []
    
    for t in range(N_t):
        flattened = an_final[t, :, 0].flatten()
        indices = np.where(flattened == 0)[0]
        if indices.size == 0:
            result.append(None)
        else:
            result.append(indices[0])

    for t in range(N_t):
        if 0 < t < N_t-1 and result[t] is not None:
            if result[t+1] is not None and result[t-1] is not None:
                x1, x2 = X[t+1, result[t+1], 1], X[t-1, result[t-1], 1]
                t1, t2 = X[t+1, result[t+1], 0], X[t-1, result[t-1], 0]         
                list_k.append(((x1-x2)/(t1-t2)).item())
            else:
                list_k.append(None)
        else:
            list_k.append(None)
    return result, list_k        


def deal_with_an_new(X, an_final):
    N_t, N_x, _ = an_final.shape
    X = X.reshape((N_t, N_x, 2))
    result = []
    list_k = []
    
    for t in range(N_t):
        res_t = []
        flattened = an_final[t, :, 0].flatten()
        x_index = 0
        while x_index < N_x:
            if np.abs(flattened[x_index]) < 1e-3:
                res_t.append(x_index)
                x_index += 5
            else:
                x_index += 1
        result.append(res_t)

    for t in range(N_t):
        len_t = len(result[t])
        res_k = []
        if 0 < t < N_t-1 and len_t != 0:
            len_m, len_p = len(result[t-1]), len(result[t+1])
            if len_m == len_t and len_p == len_t:
                for j in range(len_t):
                    x1, x2 = X[t+1, result[t+1][j], 1], X[t-1, result[t-1][j], 1]
                    t1, t2 = X[t+1, result[t+1][j], 0], X[t-1, result[t-1][j], 0]         
                    res_k.append(((x1-x2)/(t1-t2)).item())
        list_k.append(res_k)
    return result, list_k