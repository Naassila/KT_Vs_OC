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

tNode = [[0, 0.8], [0, 0.3]]
qNode = [np.deg2rad([30, -70])]

cost_functions = 'Minimize jerk, energy and time'
solutions = []
data_cost_smooth = []
data_cost_discrete = []
for i, itime in enumerate(tNode):
    n_joint = len(qLow)
    problems = []
    solution_cost = []
    for ijoint in range(n_joint):
        for ia in [5000, 10**5]:
            problems.append(set_problem(qLow[ijoint],qUpp[ijoint],
                                        dqMax[ijoint], tNode,
                                        qNode[ijoint], nGrid))
            n = 15
            d = itime
            isol = problems[ijoint].solve(cost_functions, n, d, ia)
            solution_cost.append(isol)
            dsol = pd.DataFrame({'Time':isol[3],
                                 'q':(L1*isol[4][0]-L1*isol[4][0][0])/(L1*np.diff(qNode).flatten()[0]),#np.rad2deg(isol[4][0]),
                                 'dq':L1*isol[5][0],
                                 })
            dsol = dsol.assign(MT=f'{itime[-1]} s').assign(Joint=ijoint).assign(alpha_beta = ia)
            data_cost_smooth.append(dsol)

    solutions.append(solution_cost)

# Plot results

data_smooth = pd.concat(data_cost_smooth)
df_smooth = data_smooth.melt(id_vars = ['Time', 'MT', 'Joint', 'alpha_beta'], var_name = 'which', value_name='Amplitude')
df_smooth.rename(columns={'alpha_beta':'α/β', 'MT':'$t_{end}$'}, inplace=True)
df_smooth['α/β'] = ['{:.0e}'.format(i) for i in df_smooth['α/β']]
g = sns.relplot(df_smooth, x='Time', y='Amplitude', hue='α/β', style='$t_{end}$', row='which', kind= 'line',
                row_order=['q', 'dq'], facet_kws={'sharey': False, 'sharex': True}, palette=['#b2df8a', '#33a02c'], #'Paired' ,
                height = 2, aspect = 1.8)
var_list = ['Normalized displacement', 'Velocity [m.s⁻¹]']
for i, ax in enumerate(g.axes.flatten()):
    if i == 0:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])

    else:
        ax.set_ylim(-3.8, 0)
        ax.set_yticks([-3.8, -1.9, 0])

    ax.set_title('')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(var_list[i])
    ax.set_xlim(0, 0.8)
    ax.set_xticks([0, 0.3, 0.8])
    ax.axvline(0.3, color='k', linestyle=':', alpha=0.5)
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.ticklabel_format(style='sci', axis='y')
    ax.spines[['right', 'top']].set_visible(True)



plt.savefig('./Paper/alpha_beta_effect.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)




