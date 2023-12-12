""""
Given initial and final state, and problem constraints, find the
trajectory that solve the problem
"""
import numpy as np
from scipy.optimize import *
import pandas as pd
import seaborn as sns
from Tools_constraint import *
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
L2 = 0.28 #m
nGrid = [15, 15, 15] # Order of interpolating polynoms

# Set problem
qLow = np.deg2rad([-110, 5])
qUpp = np.deg2rad([30, 140])
dqMax = [20, 20] #Speed limit

tNode = [[0, 0.05, 0.2, 0.3], [0.272, 0.8]]
qNode = np.deg2rad([[30, -10, -60, -70],
                     [60, 90, 60, 10]])

cost_functions = [
    'Minimize jerk',
    'Minimize jerk and time',
    'Minimize jerk, energy and time',
    # 'Lognormal profile'
]
solutions = []
data_cost_smooth = []
data_cost_discrete = []
joint_output_smooth = []
joint_output_discrete = []
for i, icost in enumerate(cost_functions[:]):
    n_joint = len(qLow)
    problems = []
    solution_cost = []
    for ijoint in range(n_joint):
        problems.append(set_problem(qLow[ijoint] ,qUpp[ijoint],
                                    dqMax[ijoint], tNode,
                                    qNode[ijoint], nGrid))
        n = nGrid
        d = tNode[0]
        isol = problems[ijoint].solve(icost, n, d, 100000,  )
        solution_cost.append(isol)
        dsol = pd.DataFrame({'Time':isol[3],
                             'q':isol[4][0],#np.rad2deg(isol[4][0]),
                             'dq':isol[5][0],
                             'ddq':isol[6][0]})
        dsol = dsol.assign(Cost=icost).assign(Joint=ijoint)

        if ijoint==1:
            velocity = pd.DataFrame({'Time': isol[3]})
            velocity['x_elbow'] = L1 * np.cos(data_cost_smooth[2 * i].q)
            velocity['y_elbow'] = L1 * np.sin(data_cost_smooth[2 * i].q)
            velocity['Pelbow_qdot'] = L1 * data_cost_smooth[2*i].dq
            velocity['Wrist_qdot'] = velocity.Pelbow_qdot + L2 * dsol.dq
            velocity['x_wrist'] = velocity['x_elbow'] + L2 * np.cos(data_cost_smooth[2 * i].q + dsol.q)
            velocity['y_wrist'] = velocity['y_elbow'] + L2 * np.sin(data_cost_smooth[2 * i].q + dsol.q)
            velocity.drop([100, 201], inplace=True)
            velocity['elbow_xydot'] = np.sqrt(
                np.gradient(velocity['x_elbow'], velocity.Time) ** 2 + np.gradient(velocity['y_elbow'],
                                                                                   velocity.Time) ** 2)
            velocity['Wrist_xydot'] = np.sqrt(
                np.gradient(velocity['x_wrist'], velocity.Time) ** 2 + np.gradient(velocity['y_wrist'],
                                                                                   velocity.Time) ** 2)
            joint_output_smooth.append(velocity.assign(Cost=icost))
        data_cost_smooth.append(dsol)

        dsol = pd.DataFrame({'Time': isol[0],
                             'q': isol[1], #np.rad2deg(isol[1]),
                             'dq': isol[2], })
        dsol = dsol.assign(Cost=icost).assign(Joint=ijoint)
        data_cost_discrete.append(dsol)
        # output contains [t, grid_q, grid_dq, t_interp, q, dq, ddq, dddq, cost]
    solutions.append(solution_cost)

# Plot angular positions and velocities:
data_smooth = pd.concat(data_cost_smooth)
data_smooth.q = np.rad2deg(data_smooth.q)
df_smooth = data_smooth.melt(id_vars = ['Time', 'Cost', 'Joint'], var_name = 'which', value_name='Amplitude')
# df_smooth.Amplitude = np.rad2deg(df_smooth.Amplitude)
g = sns.relplot(df_smooth, x='Time', y='Amplitude', hue='Cost', row='which', kind= 'line',col='Joint',
                row_order=['q', 'dq'], facet_kws={'sharey': False, 'sharex': True},
                height = 2, aspect = 1.5)
var_list = ['Angular position [°]', 'Angular Velocity [rad/s]']
# g.axes[0][0].set_ylim(-70, 30)
# g.axes[0][0].set_yticks([-70, -20, 30])
g.axes[0][0].set_ylabel(var_list[0])
g.axes[0][0].set_title('Proximal joint')

# g.axes[0][1].set_ylim(10, 100)
# g.axes[0][1].set_yticks([10, 55, 100])
g.axes[0][1].set_title('Distal joint')

# g.axes[1][0].set_ylim(-15, 0)
# g.axes[1][0].set_yticks([-14, -7, 0])
g.axes[1][0].set_ylabel(var_list[1])
g.axes[1][0].set_title('')

# g.axes[1][1].set_ylim(-15, 8)
# g.axes[1][1].set_yticks([-14, -3, 8])
g.axes[1][1].set_title('')
for i, ax in enumerate(g.axes.flatten()):
    # ax.set_title('')
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0, 0.3)
    ax.axvline(0.1, color='k', linestyle=':', alpha=0.5)
    ax.axvline(0.2, color='k', linestyle=':', alpha=0.5)
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.ticklabel_format(style='sci', axis='y')
    ax.spines[['right', 'top']].set_visible(True)

plt.savefig('./Paper/2DOF_constrained_joints_2.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.close()
# Plot end(mid) effectors
data_smooth = pd.concat(joint_output_smooth)
data_smooth = data_smooth[['Time', 'elbow_xydot', 'Wrist_xydot', 'Cost']]
data_smooth.rename(columns={'elbow_xydot':'Elbow', 'Wrist_xydot':'Wrist'}, inplace=True)
df_smooth = data_smooth.melt(id_vars = ['Time', 'Cost'], var_name = 'which', value_name='Amplitude')
g2 = sns.relplot(df_smooth, x='Time', y='Amplitude', hue='Cost', col='which', kind= 'line',
                col_order=['Elbow', 'Wrist'], facet_kws={'sharey': False, 'sharex': True},
                height = 2, aspect = 1.5)
var_list = ['Velocity [m/s]']
g2.axes[0][0].set_ylabel(var_list[0])
for i, ax in enumerate(g2.axes.flatten()):
    ax.set_title(ax.get_title().split()[-1])
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0, 0.3)
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.ticklabel_format(style='sci', axis='y')
    ax.spines[['right', 'top']].set_visible(True)

plt.savefig('./Paper/2DOF_constrained_effectors_2.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.close()
# Plot arm in time
data_smooth = pd.concat(joint_output_smooth)
data_smooth = data_smooth[['Time', 'x_elbow', 'y_elbow', 'x_wrist','y_wrist', 'Cost']]
sns_data = pd.concat([
    data_smooth[['Time', 'x_elbow', 'y_elbow','Cost']].rename(columns={'x_elbow':'x', 'y_elbow':'y'}).assign(Joint='Elbow'),
    data_smooth[['Time', 'x_wrist', 'y_wrist','Cost']].rename(columns={'x_wrist':'x', 'y_wrist':'y'}).assign(Joint='Wrist')])
g3 = sns.relplot(data = sns_data, x='x', y='y', hue='Cost', style='Joint', height=3, aspect=1, markers=['4', '4'], color='grey', alpha=0.5,
                 col_order=[ 'Minimize jerk', 'Minimize jerk and time',   'Minimize jerk, energy and time',], legend=False)
data_smooth.Time = data_smooth.Time.round(5)
data_smooth = data_smooth.loc[
    (data_smooth.Time ==0) |
    (data_smooth.Time ==0.100) |
    (data_smooth.Time ==0.2) |
    (data_smooth.Time ==0.3)]

data = data_smooth[data_smooth.Cost =='Minimize jerk, energy and time']
ax = g3.axes[0][0]
colors_cost = ['#1f77b4', '#ff7f0e', '#2ca02c',]

for iirow, irow in data.iterrows():
    ax.plot([0, irow.x_elbow, irow.x_wrist], [0, irow.y_elbow, irow.y_wrist], color='k', alpha = 0.4 + iirow*0.2/104 )

ax.set_title('')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xlim(-0.1, 0.8)
ax.set_ylim(-0.6, 0.6)
ax.set_aspect('equal')
ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax.tick_params(axis='y', which='both', left=False, labelleft=False)
ax.spines[['right', 'top']].set_visible(True)

plt.savefig('./Paper/2DOF_constrained_in_time_2.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)



