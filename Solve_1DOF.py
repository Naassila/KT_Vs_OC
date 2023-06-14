""""
Given initial and final state, and problem constraints, find the
trajectory that solve the problem
"""
import numpy as np
from scipy.optimize import *
import pandas as pd
import seaborn as sns
from Tools import *
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import axes3d

params = {'legend.fontsize': 'medium',
          'axes.labelsize': 'medium',
          'axes.titlesize':'medium',
          'xtick.labelsize':'medium',
          'ytick.labelsize':'medium'}

plt.rcParams.update(params)
plt.rcParams['font.serif'] ="Times New Roman"
plt.rcParams['font.family'] ="serif"
plt.rcParams.update({'font.size': 10})
sns.set_context("paper", font_scale=1.3)

L1 = 0.3 #m
nGrid = [15] # Order of interpolating polynoms

# Set problem
qLow = np.deg2rad([-110])
qUpp = np.deg2rad([30])
dqMax = [20] #Speed limit

tNode = [0, 0.8]
qNode = [np.deg2rad([30, -70])]

cost_functions = ['Minimize jerk', 'Minimize jerk and time', 'Minimize jerk, energy and time', 'Lognormal profile']
solutions = []
data_cost_smooth = []
data_cost_discrete = []
for i, icost in enumerate(cost_functions[:-1]):
    n_joint = len(qLow)
    problems = []
    solution_cost = []
    for ijoint in range(n_joint):
        problems.append(set_problem(qLow[ijoint] ,qUpp[ijoint],
                                    dqMax[ijoint], tNode,
                                    qNode[ijoint], nGrid))
        n = 15
        d = [0, 0.8]
        isol = problems[ijoint].solve(icost, n, d, 100000)
        solution_cost.append(isol)
        dsol = pd.DataFrame({'Time':isol[3],
                             'q':np.rad2deg(isol[4][0]),
                             'dq':L1*isol[5][0],
                             'ddq':isol[6][0]})
        dsol = dsol.assign(Cost=icost).assign(Joint=ijoint)
        data_cost_smooth.append(dsol)
        dsol = pd.DataFrame({'Time': isol[0],
                             'q': np.rad2deg(isol[1]),
                             'dq': L1*isol[2], })
        dsol = dsol.assign(Cost=icost).assign(Joint=ijoint)
        data_cost_discrete.append(dsol)
        # output contains [t, grid_q, grid_dq, t_interp, q, dq, ddq, dddq, cost]
    solutions.append(solution_cost)

# Plot results

D = np.trapz(data_cost_smooth[0].Time, data_cost_smooth[2].dq.values)
y1 = -data_cost_smooth[2].dq.values/D
param_log = home_fit_lognpdf(data_cost_smooth[2].Time, data_cost_smooth[2].dq.values, 0.3, 0.4,
                     [-2.2, 0.1, 0], [-1.5, 0.6, 1e-2])
param_log = inflexion_points_logparam(data_cost_smooth[2].Time, data_cost_smooth[2].dq.values,
                                      data_cost_smooth[2].ddq.values,[-1.5, 0.6, 1e-2])
params, extras = curve_fit(lognpdf, data_cost_smooth[0].Time, y1, bounds=((-2.2, 0.1, 0), (-1.5, 0.6, 1e-2)))
v_TK = pd.DataFrame({'Time': data_cost_smooth[0].Time,
                     # 'dq': -D*lognpdf(data_cost_smooth[0].Time.values,
                     #                       params[0], params[1],
                     #                       t0 = 0),
                     'dq': -param_log[2]*lognpdf(data_cost_smooth[0].Time.values,
                                           param_log[0], param_log[1],
                                           t0 = param_log[3]),
                     }
                    ).assign(Cost=cost_functions[-1])

data_discrete = pd.concat(data_cost_discrete)
data_cost_smooth.extend([v_TK])
data_smooth = pd.concat(data_cost_smooth)
df_smooth = data_smooth.melt(id_vars = ['Time', 'Cost', 'Joint'], var_name = 'which', value_name='Amplitude')
df_discrete = data_discrete.melt(id_vars = ['Time', 'Cost', 'Joint'], var_name = 'which', value_name='Amplitude')
g = sns.relplot(df_smooth, x='Time', y='Amplitude', hue='Cost', row='which', kind= 'line',
                row_order=['q', 'dq'], facet_kws={'sharey': False, 'sharex': True},
                height = 2, aspect = 2)
var_list = ['Angular position [°]', 'Velocity [m.s⁻¹]']
for i, ax in enumerate(g.axes.flatten()):
    if i == 0:
        ax.set_yticks([30, -20, -70])
        sns.scatterplot(data = data_discrete, x='Time', y='q', ax=ax,
                        hue='Cost', s=13, legend=False)
    else:
        sns.scatterplot(data=data_discrete, x='Time', y='dq',
                        hue='Cost', ax=ax,  s=13, legend=False)
    #     ax.set_yticks([30, -20, -70])
    ax.set_title('')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(var_list[i])
    ax.set_xlim(0, 0.8)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.ticklabel_format(style='sci', axis='y')
    ax.spines[['right', 'top']].set_visible(True)
    plt.savefig('./Paper/OC_Vs_KT.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)




