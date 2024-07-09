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
from Arm_1DOF import main

params = {'legend.fontsize': 'medium',
          'axes.labelsize': 'medium',
          'axes.titlesize':'medium',
          'xtick.labelsize':'medium',
          'ytick.labelsize':'medium'}

plt.rcParams.update(params)
plt.rcParams['font.serif'] ="Times New Roman"
plt.rcParams['font.family'] ="serif"
plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
plt.rcParams.update({'font.size': 10})
sns.set_context("paper", font_scale=1.3)

t_end = [0.8, 0.4]

cost_functions = 'Minimize jerk, energy and time'

data_cost_smooth = []
t0 = 0.01
for i, itime in enumerate(t_end):

    for ia in [5000, 10**5]:
        isol = main(n_shooting = 200,
                    final_time=itime,
                    cost_function_index=[2],
                    alpha_beta=ia,
                    t0=t0,
                    inside_plot=False,
                    external_use=True)

        data_for_snr = isol[0][['Time', 'Elbow_vel', 'Elbow_acc', 'cost']]
        t = data_for_snr.Time.values
        param_log_snr = inflexion_points_logparam_robust_t0_fixed(t, data_for_snr.Elbow_vel.values,
                                                                  np.gradient(data_for_snr.Elbow_vel.values, t),
                                                                  [0.1, 0.6, 0.1],
                                                                  t0=t0-t0)
        log_for_cost = param_log_snr[2][0]*lognpdf(t,param_log_snr[0][0], param_log_snr[1][0],t0 = param_log_snr[3][0])
        snr_value = snr(data_for_snr.Elbow_vel.values, log_for_cost)
        print(f'For {ia}, SNR = {snr_value}')
        log_output = f'The lognormal is defined with D:{np.round(param_log_snr[2][0], 2)}, mu:' \
                     f'{np.round(param_log_snr[0][0], 2)},sigma' \
                     f':{np.round(param_log_snr[1][0], 2)} and t0: {np.round(param_log_snr[3][0], 3)}'
        print(log_output)

        data_cost_smooth.append(isol[0].assign(MT=f'{itime} s').assign(alpha_beta=ia))

# Plot results
import matplotlib.ticker as mticker
ff = mticker.ScalarFormatter(useOffset=False, useMathText=True)
ff.set_powerlimits((-5,5))

data_smooth = pd.concat(data_cost_smooth)
data_smooth.drop(['Elbow_x', 'Elbow_y', 'Elbow_acc'], axis=1, inplace=True)
data_smooth.rename(columns={'Elbow_d':'q', 'Elbow_vel':'dq'}, inplace=True)
df_smooth = data_smooth.melt(id_vars = ['Time', 'MT', 'alpha_beta'], var_name = 'which', value_name='Amplitude')
df_smooth.rename(columns={'alpha_beta':'α/β', 'MT':'$t_{end}$'}, inplace=True)
# df_smooth['α/β'] = ['{:.0e}'.format(i) for i in df_smooth['α/β']]
df_smooth['α/β'] = ['${}$'.format(ff.format_data(i)) if i==10**5 else i for i in df_smooth['α/β'] ]
g = sns.relplot(df_smooth, x='Time', y='Amplitude', hue='α/β', style='$t_{end}$', row='which', kind= 'line',
                row_order=['q', 'dq'], facet_kws={'sharey': False, 'sharex': True}, palette=['#33a02c', '#703c9c'], ##b2df8a'], #'Paired' ,
                height = 2, aspect = 1.8)
var_list = ['Displacement norm [m]', 'Velocity norm [m.s⁻¹]']


# g.legend.get_texts()[2].set_text('${}$'.format(ff.format_data(float(g.legend.get_texts()[2].get_text()))))

for i, ax in enumerate(g.axes.flatten()):
    if i == 0:
        ax.set_ylim(0, 0.5236)
        ax.set_yticks([0, 0.25, 0.5])

    else:
        ax.set_ylim(-0.1, 3.1)
        ax.set_yticks([0, 1.5, 3])

    ax.set_title('')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(var_list[i])
    ax.set_xlim(0, 0.8)
    ax.set_xticks([0, 0.4, 0.8])
    ax.axvline(0.4, color='k', linestyle=':', alpha=0.5)
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.ticklabel_format(style='sci', axis='y')
    ax.spines[['right', 'top']].set_visible(True)



plt.savefig('./Paper/Biotim_alpha_beta_effect.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)




