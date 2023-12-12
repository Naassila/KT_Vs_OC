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
from article_Najoua_1DOF import main

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

t_end = [0.8, 0.3]

cost_functions = 'Minimize jerk, energy and time'

data_cost_smooth = []

for i, itime in enumerate(t_end):

    for ia in [5000, 10**5]:
        isol = main(n_shooting = 120,
                    final_time=itime,
                    cost_function_index=[2],
                    alpha_beta=ia,
                    inside_plot=False,
                    external_use=True)

        data_cost_smooth.append(isol.assign(MT=f'{itime} s').assign(alpha_beta = ia))

            # dsol = pd.DataFrame({'Time':isol[3],
            #                      'q':(L1*isol[4][0]-L1*isol[4][0][0])/(L1*np.diff(qNode).flatten()[0]),#np.rad2deg(isol[4][0]),
            #                      'dq':L1*isol[5][0],
            #                      })


# Plot results

data_smooth = pd.concat(data_cost_smooth)
data_smooth.drop(['Elbow_x', 'Elbow_y', 'Elbow_acc'], axis=1, inplace=True)
data_smooth.rename(columns={'Elbow_d':'q', 'Elbow_vel':'dq'}, inplace=True)
df_smooth = data_smooth.melt(id_vars = ['Time', 'MT', 'alpha_beta'], var_name = 'which', value_name='Amplitude')
df_smooth.rename(columns={'alpha_beta':'α/β', 'MT':'$t_{end}$'}, inplace=True)
df_smooth['α/β'] = ['{:.0e}'.format(i) for i in df_smooth['α/β']]
g = sns.relplot(df_smooth, x='Time', y='Amplitude', hue='α/β', style='$t_{end}$', row='which', kind= 'line',
                row_order=['q', 'dq'], facet_kws={'sharey': False, 'sharex': True}, palette=['#b2df8a', '#33a02c'], #'Paired' ,
                height = 2, aspect = 1.8)
var_list = ['Displacement norm [m]', 'Velocity norm [m.s⁻¹]']

for i, ax in enumerate(g.axes.flatten()):
    if i == 0:
        ax.set_ylim(0, 0.5236)
        ax.set_yticks([0, 0.25, 0.5])

    else:
        ax.set_ylim(-0.1, 3.9)
        ax.set_yticks([0, 1.9, 3.8])

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



plt.savefig('./Paper/Biotim_alpha_beta_effect.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)




