import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Get the path of the current file and add its grandparent directory to sys.path
# so that `import utils` works when this file is run directly.
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils


config_ts = {"temporal_range": [0.0, 6.0], "spatial_range": [-4.0, 6.0]}


def f(u):
    return u * (5-u)

def df(u):
    return 5 - 2*u

def solution(arr_t, arr_x):
    g_S1 = lambda t: t + 2
    g_S2 = lambda t: -2 - 3*t + 8*np.sqrt(t)
    g_S3 = lambda t: -t + 6
    g_R1 = lambda t: t - 2
    g_R2 = lambda t: 5*t - 2
    func1 = lambda x, t: np.select(
        [(x < g_R1(t)), ((x >= g_R1(t)) & (x < g_R2(t))), 
         ((x >= g_R2(t)) & (x < g_S1(t))), (x >= g_S1(t))], 
        [2, (5-(x+2)/t)/2, 0, 4]
    )
    func2 = lambda x, t: np.select(
        [(x < g_R1(t)), ((x >= g_R1(t)) & (x < g_S2(t))), (x >= g_S2(t))], 
        [2, (5-(x+2)/t)/2, 4]
    )
    func3 = lambda x, t: np.where(
        x < g_S3(t), 2, 4
    )
    func = lambda x, t: np.select(
        [(t < 1), ((t >= 1) & (t <= 4)), (t > 4)], 
        [func1(x, t), func2(x, t), func3(x, t)]
    )
    return func(arr_x, arr_t)


if __name__ == "__main__":
    xmin, xmax = config_ts["spatial_range"]
    tmin, tmax = config_ts["temporal_range"]
    N_t, N_x = 64, 512
    h = (xmax-xmin) / (N_x-1)

    x_values = np.linspace(xmin, xmax, N_x)
    t_values = np.linspace(tmin, tmax, N_t)

    t_grid, x_grid = np.meshgrid(t_values, x_values)

    t_final = t_grid.flatten()
    x_final = x_grid.flatten()
    u_final = solution(t_final, x_final).reshape(-1, 1)
    an_final = utils.AN(df(solution(t_final, x_final-h)), df(solution(t_final, x_final+h)), h)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    axs[0].scatter(x_final, t_final, c=u_final, s=2, rasterized=True)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('t')
    axs[0].set_title('Reference')

    axs[1].scatter(x_final, t_final, c=an_final, s=2, rasterized=True)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('t')
    axs[1].set_title('AN Indicator')

    plt.tight_layout()
    plt.savefig("Figures/fig_case/indicator_2B.eps", dpi=200)