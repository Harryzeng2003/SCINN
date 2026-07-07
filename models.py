# Import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import pdb
sys.path.insert(1, "../../")
import utils
from scipy.optimize import fsolve



class SCINN(nn.Module):
	def __init__(self, config, umax, umin, num_hidden_layers=4, size_hidden=100):
		"""Class for the model.
		
		Parameters
		----------
		config : dictionary
			Configuration file for the training.
		num_hidden_layers : int
			Number of hidden layers (default: 4)
		size_hidden : int
			Number of neurons per hidden layer (default: 100)
		"""
		
		super(SCINN, self).__init__()
		self.config = config
		self.N_t = eval(self.config["physical"]["N_t"])
		self.N_x = eval(self.config["physical"]["N_x"])
		self.size_hidden = size_hidden
		self.num_hidden_layers = num_hidden_layers
		self.DTYPE, self.device = eval(self.config["training_process"]["DTYPE"]), torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.num_inputs, self.num_outputs = 2, 1
		self.umax, self.umin = umax, umin
		self.Lambda = torch.ones((self.N_t, self.N_x, 1)).to(self.device)
		self.w_pt_EQ = torch.ones((self.N_t*self.N_x, 1)).to(self.device)
		self.w_pt_IF = torch.ones((self.N_t*self.N_x, 1)).to(self.device)
		self.is_PRE = False
		self.is_WE = False
		self.epoch = -1
		# Hyperparameters (overridden by training_script from config)
		self.N_AN = 500  # Epoch interval for re-computing the discontinuity indicator
		self.eta_IF = 50  # Multiplier for implicit-formula point reweighting
		self.alpha = 0.1  # WE denominator parameter
		self.beta = 1.0   # WE exponent parameter

		# Define the DNN.
		self.dense_layers = []
		self.dense_layers.append(nn.Linear(self.num_inputs, self.size_hidden).to(self.device))
		for i in range(0, self.num_hidden_layers):
			layer = nn.Linear(self.size_hidden, self.size_hidden).to(self.device)
			self.dense_layers.append(layer)
		layer_final = nn.Linear(self.size_hidden, self.num_outputs).to(self.device)
		self.dense_layers.append(layer_final)

		# Initialize the weights of the layers.
		for i in range(len(self.dense_layers)):
			nn.init.xavier_uniform_(self.dense_layers[i].weight, gain=5.0/3)

		# Now, register the parameters as trainable variables
		self.params_hidden = nn.ModuleList(self.dense_layers)
		
		# Related with the L2 computation.
		self.l2_EQ_hist = []      # Equation residual L2 history
		self.l2_IBC_hist = []      # Initial/boundary condition L2 history
		self.l2_AC_hist = []      # Exact solution L2 history (away from discontinuity)
		self.l2_IF_hist = []      # Implicit formula L2 history
		self.l2_RH_hist = []      # Rankine-Hugoniot (shock) L2 history
		self.l2_all_hist = []     # Total loss history

		# Define activation functions.
		self.act_ux = eval(self.config["neural"]["activation_functions"]["output"][0])
		self.act_hidden = eval(self.config["neural"]["activation_functions"]["hidden_layers"])

	def v_2_vtilde(self, v, vmax, vmin):
		return (v - vmin) / (vmax - vmin)
	
	def vtilde_2_v(self, vtilde, vmax, vmin):
		return vtilde * (vmax - vmin) + vmin
	
	def forward(self, X):
		# Training bucle.
		X = self.act_hidden(self.dense_layers[0](X))
		for i in range(1, len(self.dense_layers)-1):
			X = X + self.act_hidden(self.dense_layers[i](X))
		X = self.dense_layers[-1](X)

		# Extract each primitive variable separately.
		ux = self.act_ux(X[:,0:1])
		return ux

	def compute_loss(self, X, X_0, U_0, U_exact, is_RAR, f, df, list_w, bc_solu):
		w_E, w_I, w_U, w_S = list_w
		num_res, num_pred = 500, 500

		t, x = X[:,0:1], X[:,1:2]
		T_max = torch.max(t).item() if self.is_PRE else 1.0
		X_max = torch.max(x).item() if self.is_PRE else 1.0
		X_min = torch.min(x).item() if self.is_PRE else 0.0
		t_tilde = self.v_2_vtilde(t, T_max, 0.0)
		x_tilde = self.v_2_vtilde(x, X_max, X_min)

		prediction = self(torch.cat((t_tilde, x_tilde), dim=1))

		#ux = prediction[:,0:1]
		ux_tilde = prediction[:,0:1]
		ux = self.vtilde_2_v(ux_tilde, self.umax, self.umin)
		if self.is_PRE:
			u_t = torch.autograd.grad(ux_tilde, t_tilde, grad_outputs=torch.ones_like(ux_tilde), create_graph=True)[0]
			ux_x = torch.autograd.grad(ux_tilde, x_tilde, grad_outputs=torch.ones_like(ux_tilde), create_graph=True)[0]
			residual = (u_t / T_max + df(ux) * ux_x / (X_max - X_min))
		else:
			u_t = torch.autograd.grad(ux, t, grad_outputs=torch.ones_like(ux_tilde), create_graph=True)[0]
			ux_x = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux_tilde), create_graph=True)[0]
			residual = (u_t + df(ux) * ux_x)
		if self.is_WE:
				self.Lambda = (1/(self.alpha*abs(ux_x)**self.beta+1)).view(self.N_t, self.N_x, 1)

		# Compute Losses
		# ================================================================================================================================
		# Losses of the equations conforming the system
		L_t = torch.square(residual).view(self.N_t, self.N_x, 1)
		L_t_ga = torch.mean(L_t, dim=1)[1:].mean()
		L_t = self.Lambda * L_t
		if is_RAR:
			_, indices = torch.topk(L_t.view(self.N_t * self.N_x), num_res, largest=True)
			self.w_pt_EQ[indices, :] *= 5
		L_t = torch.mean(L_t, dim=1)[1:].mean()
		L_t_w = torch.mean(self.w_pt_EQ.view(self.N_t, self.N_x, 1) * L_t, dim=1)[1:].mean()
		# ================================================================================================================================
		
		# Compute loss for tmin (L_IC).
		t0, x0 = X_0[:,0:1], X_0[:,1:2]
		x0_tilde = self.v_2_vtilde(x0, X_max, X_min)

		prediction_tmin_tilde = self(torch.cat((t0, x0_tilde), dim=1))
		prediction_tmin = self.vtilde_2_v(prediction_tmin_tilde, self.umax, self.umin)

		# Consider a certain weight for the IC (hyperparameter) and for the collocation loss.
		# w_ux = self.config["neural"]["loss_function_parameters"]["w_IC"]

		# Compute initial losses.
		L_IC_ux = torch.square(U_0[:,0:1] - prediction_tmin[:,0:1]).mean()

        # Compute boundary losses.
		L_BC_ux = 0
		pred_xt = ux.reshape((self.N_t, self.N_x))
		pred_L, pred_R = pred_xt[:, 0].reshape(-1,1), pred_xt[:, -1].reshape(-1,1)
		# time = np.linspace(self.tmin, self.tmax, self.N_t).reshape(-1,1)
		# x_L = np.linspace(self.xmin, self.xmin, self.N_t).reshape(-1,1)
		# x_R = np.linspace(self.xmax, self.xmax, self.N_t).reshape(-1,1)
		u_L = utils.initial_conditions(X, torch.tensor(self.xmin), self.case, self.config)
		u_R = utils.initial_conditions(X, torch.tensor(self.xmax), self.case, self.config)
		L_BC_ux = L_BC_ux + torch.square(pred_L - u_L).mean() / 2.0
		L_BC_ux = L_BC_ux + torch.square(pred_R - u_R).mean() / 2.0
		
		if w_U > 0:
			L_BC_ux *= 0.01

		l2_ga = torch.square(U_exact - ux).view(self.N_t, self.N_x, 1)

		l2_ux = torch.mean(l2_ga, dim=1).mean()
		l2_ga = torch.mean(l2_ga, dim=1).mean()
		
		l2_ifnn_w = torch.tensor(0.0).to(self.device)
		if w_U > 1e-3:
			l2_ifnn = utils.initial_conditions(X, x-df(ux)*t, self.case, self.config).to(self.device)
			l2_ifnn = (l2_ifnn - ux) # * Lambda.view(self.N_t * self.N_x, 1)
			if is_RAR:
				_, indices = torch.topk(l2_ifnn.view(self.N_t * self.N_x), num_pred, largest=True)
				self.w_pt_IF[indices, :] *= self.eta_IF
			l2_ifnn = torch.sqrt(torch.square(l2_ifnn).sum() / x.numel()).mean()		
			l2_ifnn_w = torch.sqrt(torch.square(self.w_pt_IF * l2_ifnn).sum() / x.numel()).mean()


		l2_S = torch.tensor(0.0).to(self.device)
		ux_square = ux.view(self.N_t, self.N_x, 1)
		
		if w_S > 1e-3:
			nums = []
			# Use AN indicator
			if self.epoch == 0:
				self.list_k = [[] for _ in range(self.N_t)]
			elif (self.epoch >= 2500 and self.epoch % self.N_AN == 0):
				ux_left = ux_square[:, :-2, :].cpu().detach().numpy()
				ux_right = ux_square[:, 2:, :].cpu().detach().numpy()
				temp = utils.AN(df(ux_left), df(ux_right), self.dx)
				self.an_res = np.pad(temp, pad_width=((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=1)
				self.indexes, self.list_k = utils.deal_with_an_new(X, self.an_res)
				
			for j in range(1, self.N_t-1):
				num_k = self.list_k[j]
				for i in range(len(num_k)):
					u_l, u_r = ux_square[j, self.indexes[j][i]-1, 0], ux_square[j, self.indexes[j][i]+1, 0]
					nums.append(((f(u_l) - f(u_r)) / (u_l - u_r) - num_k[i]) ** 2)
			if len(nums) != 0:
				l2_S = torch.mean(torch.stack(nums))
			else:
				l2_S = torch.tensor(0.0).to(self.device)
		loss = w_E * L_t_w + w_I * (L_IC_ux + L_BC_ux) + w_U * l2_ifnn_w + w_S * l2_S

		# Take advantage and save initial losses into lists
		# self.loss_ic_hist.append(L_IC_ux.item())
		# self.loss_ic_ux.append(torch.square(U_0[:,0:1] - prediction_tmin[:,0:1]).mean().item())
		self.l2_EQ_hist.append(L_t_w.item())
		self.l2_IBC_hist.append((L_IC_ux+L_BC_ux).item())
		self.l2_AC_hist.append(l2_ux.item())
		self.l2_IF_hist.append(l2_ifnn_w.item())
		self.l2_RH_hist.append(l2_S.item())
		self.l2_all_hist.append(loss.item())

		# Compute and save l2 errors
		# self.compute_l2()
		if is_RAR:
			return loss, torch.abs(residual)
		else:
			return loss, torch.abs(residual)



	def train_step(self, X, X_0, U_0, U_exact, optimizer, scheduler, is_RAR, f, df, list_w, bc_solu):
		optimizer.zero_grad(set_to_none=True)
		
		# Compute the loss
		loss, res = self.compute_loss(X, X_0, U_0, U_exact, is_RAR, f, df, list_w, bc_solu)
		loss.backward(retain_graph=False)

		# Apply gradient clipping and update parameters
		# torch.nn.utils.clip_grad_value_(self.parameters(), 2.5)
		# torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
		optimizer.step()
		scheduler.step()

		# Save data
		
		return loss.item(), res
