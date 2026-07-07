# Import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import json
import utils
import models

import Cases.solu_case5 as solu

# Load configuration file
# -------------------------------------------------------------------------
# Config sections:
#   physical         - problem setup: case, N_t (time steps), N_x (space points), N_0 (initial points)
#   neural           - activation functions and training_parameters:
#       training_parameters:
#           PRE      - whether to preprocess (normalize) input coordinates [true/false]
#           WE       - whether to use weighted equation loss (Lambda weighting) [true/false]
#           NUM      - loss weights [w_E, w_I, w_U, w_S] for equation, IC/BC, implicit-formula, shock
#           N_PAR    - epoch interval for saving checkpoints and triggering RAR
#           N_AN     - epoch interval for re-computing the discontinuity indicator
#           eta_IF   - multiplier for implicit-formula point reweighting
#           alpha    - WE denominator parameter: Lambda = 1 / (alpha * |u_x|^beta + 1)
#           beta     - WE exponent parameter
#           num_hidden / num_neurons - DNN architecture (hidden layers, neurons per layer)
#   training_process - DTYPE, export paths, optimizer, learning_rate, epochs, random_seed
# -------------------------------------------------------------------------
try:
	with open("config.json") as file:
		config = json.load(file)
except:
	print("Error: The configuration file does not exist or it contains errors.")
	sys.exit(1)

case = config["physical"]["case"]
DTYPE = eval(config["training_process"]["DTYPE"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_model = "Example_Problem/" + config["training_process"]["export"]["path_models"]
path_data = "Example_Problem/" + config["training_process"]["export"]["path_data"]
path_images = config["training_process"]["export"]["path_images"]

# Ensure all output directories exist
os.makedirs(path_model, exist_ok=True)
os.makedirs(path_data, exist_ok=True)
os.makedirs(path_images, exist_ok=True)

tmin, tmax = solu.config_ts["temporal_range"]
xmin, xmax = solu.config_ts["spatial_range"]
N_t = eval(config["physical"]["N_t"])
N_x = eval(config["physical"]["N_x"])
N_0 = eval(config["physical"]["N_0"])

lr = config["training_process"]["parameters"]["learning_rate"]
epochs = config["training_process"]["parameters"]["epochs"]

# Training parameters from config
tp = config["neural"]["training_parameters"]
is_PRE = tp["PRE"]
is_WE = tp["WE"]
list_w = tp["NUM"]
N_PAR = tp["N_PAR"]
N_AN = tp["N_AN"]
eta_IF = tp["eta_IF"]
alpha = tp["alpha"]
beta = tp["beta"]
num_hidden = tp["num_hidden"]
num_neurons = tp["num_neurons"]

# Generation of data.
h = (xmax - xmin) / eval(config["physical"]["N_x"])
X, X_0 = utils.generate_domain(N_t, N_x, N_0, tmin, tmax, xmin, xmax)
X_l, X_r = X.clone(), X.clone()
X_l[:, 1] = X_l[:, 1] - h
X_r[:, 1] = X_r[:, 1] + h

T_np, X_np = X[:, 0:1].detach().numpy(), X[:, 1:2].detach().numpy()

U_exact = torch.tensor(solu.solution(T_np, X_np)).to(device)
func, dfunc = solu.f, solu.df

X, X_0 = X.to(DTYPE).to(device), X_0.to(DTYPE).to(device)
X_l, X_r = X_l.to(DTYPE).to(device), X_r.to(DTYPE).to(device)

# Define initial conditions
U_0 = utils.initial_conditions(X, X_0[:, 1:2], case, config)
umax, umin = torch.max(U_0).item(), torch.min(U_0).item()
U_0 = U_0.to(device)

# Seeds
torch.manual_seed(config["training_process"]["parameters"]["random_seed"])
np.random.seed(config["training_process"]["parameters"]["random_seed"])

model = models.SCINN(config, umax=umax, umin=umin,
                       num_hidden_layers=num_hidden,
                       size_hidden=num_neurons).to(device)
model.tmin, model.tmax = solu.config_ts["temporal_range"]
model.xmin, model.xmax = solu.config_ts["spatial_range"]
model.case = case

# Define the optimizer.
if config["training_process"]["parameters"]["optimizer"] == "RAdam":
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
elif config["training_process"]["parameters"]["optimizer"] == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
elif config["training_process"]["parameters"]["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif config["training_process"]["parameters"]["optimizer"] == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9, dampening=0)
else:
    sys.exit("Optimizer is invalid. Only the following are available at this moment:\n-'RAdam'\n-'AdamW'\n-'Adam'\n-'SGD'")
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=1.0)


# Set model hyperparameters
model.is_PRE = is_PRE
model.is_WE = is_WE
model.dx = (model.xmax - model.xmin) / (model.N_x - 1)
model.N_AN = N_AN
model.eta_IF = eta_IF
model.alpha = alpha
model.beta = beta

min_loss, min_epoch, min_res = 100000, -1, torch.zeros((model.N_t*model.N_x, 1)).to(device)
min_Lambda = torch.ones((model.N_t, model.N_x, 1)).to(device)
min_para = model.state_dict().copy()
print("------ Case %d ------" % case)

# Record training start time
import time
train_start_time = time.time()

# Reset CUDA memory statistics
if torch.cuda.is_available():
    if hasattr(torch.cuda, 'reset_max_memory_allocated'):
        torch.cuda.reset_max_memory_allocated()
    if hasattr(torch.cuda, 'reset_max_memory_reserved'):
        torch.cuda.reset_max_memory_reserved()

# RAR executes at most once
rar_executed = False

for epoch in range(epochs):
    is_save = epoch % N_PAR == 0
    # is_RAR can only be True once
    if is_WE and is_save and (epoch > 0) and not rar_executed:
        is_RAR = True
        rar_executed = True
    else:
        is_RAR = False
    model.epoch = epoch
    loss, res = model.train_step(X, X_0, U_0, U_exact, optimizer, scheduler,
                                 is_RAR, func, dfunc, list_w, solu.solution)
    loss_l2 = model.l2_AC_hist[-1]
    if epoch % 100 == 0:
        print("epoch: %d, ibc: %.3e, loss: %.3e, l2: %.3e" % (epoch, model.l2_IBC_hist[-1], loss, loss_l2))
    if epoch % 500 == 0:
        try:
            utils.plot_results(X, U_exact, model, model.Lambda, case, min_res, dfunc, path_images)
        except:
            break
    if (loss_l2 < min_loss and epoch > 1000):
        min_loss, min_epoch = loss_l2, epoch
        min_para, min_res = model.state_dict().copy(), res
        min_Lambda = model.Lambda.clone()
        torch.save(model.state_dict(), path_model + "case%d.pt" % case)


utils.save_results(model, path_data, case)

# Record training end time and compute duration
train_end_time = time.time()
train_duration = train_end_time - train_start_time

print("case-%d" % case)
print("min_epoch:", min_epoch)
print("min_l2_ac_loss: %.3e" % model.l2_AC_hist[min_epoch])
print("min_l2_eq_loss: %.3e" % model.l2_EQ_hist[min_epoch])

# Print training time (seconds)
print("Training time: %.2f seconds" % train_duration)

# Compute model memory usage
def get_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    total_mb = total_bytes / (1024 ** 2)
    return total_params, total_mb

params_count, memory_mb = get_model_memory(model)
print(f"Model parameters: {params_count:,}")
print(f"Model memory: {memory_mb:.2f} MB")

# Print peak GPU memory usage (if using GPU)
if torch.cuda.is_available():
    max_gpu_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    max_gpu_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(f"Max GPU memory allocated during training: {max_gpu_allocated:.2f} MB")
    print(f"Max GPU memory reserved during training: {max_gpu_reserved:.2f} MB")