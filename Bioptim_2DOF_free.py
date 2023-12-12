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

from casadi import MX, SX, vertcat, horzcat, gradient

from bioptim import(
BiorbdModel,
ConfigureProblem,
OptimalControlProgram,
DynamicsFcn,
DynamicsEvaluation,
DynamicsFunctions,
DynamicsList,
Dynamics,
BoundsList,
InitialGuessList,
NonLinearProgram,
ObjectiveFcn,
Objective,
ObjectiveList,
PenaltyController,
Solver,
)

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


def custom_func_comp(controller: PenaltyController,
                     marker:str,
                     alpha:float,
                     ):
    q = controller.states['q'].mx


    # return objective


L1 = 0.3 #m
L2 = 0.28 #m
R = 0.03 #m
m1 = 2.4 #kg
m2 = 1.8 #kg
I1atcenter = m1*(R**2/4+L1**2/12)
I2atcenter = m2*(R**2/4+L2**2/12)
I1atorigin = I1atcenter+m1*(L1/2)**2
I2atorigin = I2atcenter+m2*(L2/2)**2
nGrid = [15] # Order of interpolating polynoms

bio_model = BiorbdModel("arm.bioMod")
dynamics = Dynamics(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)

x_bounds = BoundsList()
x_bounds["q"] = bio_model.bounds_from_ranges("q")
x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
# x_bounds["q"][:, [0, -1]] = np.deg2rad([30, 60]) # first and last state x_bound[phase][q and qdot, nodes]
x_bounds["q"][0, [0, -1]] = np.deg2rad([30, 60])
x_bounds["q"][1, [0, -1]] = np.deg2rad([-70, 10])
x_bounds["qdot"][:, [0, -1]] = 0

# Define control path constraint
n_qddot_joints = bio_model.nb_qddot - bio_model.nb_root
qddot_joints_min, qddot_joints_max, qddot_joints_init = -1000, 1000, 0
u_bounds = Bounds([qddot_joints_min] * n_qddot_joints, [qddot_joints_max] * n_qddot_joints)

u_bounds = BoundsList()
u_bounds['qddot_joints'] = [-1000, 1000], [-1000, 1000]

objective_functions = ObjectiveList()
objective_functions.add(
    custom_func_comp,
    custom_type=ObjectiveFcn.Lagrange,
    quadratic=True,
    alpha = 10000,
    marker="COM_hand",
    # ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"
)


ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting=50,
        phase_time=0.3,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        assume_phase_dynamics=True,
    )
# ocp.check_conditioning()

solver = Solver.IPOPT(show_online_optim=False)
sol = ocp.solve(solver)

q = sol.states['q']
dq = sol.states['qdot'][1]

# Set problem
qLow = np.deg2rad([-110, 5])
qUpp = np.deg2rad([30, 140])
dqMax = [20, 20] #Speed limit

tNode = [[0, 0.3], [0.272, 0.8]]
qNode = np.deg2rad([[30, -70],
                    [60, 10]])

cost_functions = [
    'Minimize jerk', 'Minimize jerk and time',
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
        n = nGrid[0]
        d = tNode[0]
        isol = problems[ijoint].solve(icost, n, d, 100000, second_joint=ijoint, )
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
g.axes[0][0].set_ylim(-70, 30)
g.axes[0][0].set_yticks([-70, -20, 30])
g.axes[0][0].set_ylabel(var_list[0])
g.axes[0][0].set_title('Proximal joint')

g.axes[0][1].set_ylim(10, 60)
g.axes[0][1].set_yticks([10, 35, 60])
g.axes[0][1].set_title('Distal joint')

g.axes[1][0].set_ylim(-15, 0)
g.axes[1][0].set_yticks([-14, -7, 0])
g.axes[1][0].set_ylabel(var_list[1])
g.axes[1][0].set_title('')

g.axes[1][1].set_ylim(-7, 0)
g.axes[1][1].set_yticks([-6, -3, 0])
g.axes[1][1].set_title('')
for i, ax in enumerate(g.axes.flatten()):
    # ax.set_title('')
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0, 0.3)
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.ticklabel_format(style='sci', axis='y')
    ax.spines[['right', 'top']].set_visible(True)

plt.savefig('./Paper/2DOF_free_joints.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.close()
# Plot end(mid) effectors
data_smooth = pd.concat(joint_output_smooth)
data_smooth = data_smooth[['Time', 'elbow_xydot', 'Wrist_xydot', 'Cost']] # 'Pelbow_qdot', 'Wrist_qdot',
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

plt.savefig('./Paper/2DOF_free_effectors.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.close()
# Plot arm in time
data_smooth = pd.concat(joint_output_smooth)
data_smooth = data_smooth[['Time', 'x_elbow', 'y_elbow', 'x_wrist','y_wrist', 'Cost']]
sns_data = pd.concat([
    data_smooth[['Time', 'x_elbow', 'y_elbow','Cost']].rename(columns={'x_elbow':'x', 'y_elbow':'y'}).assign(Joint='Elbow'),
    data_smooth[['Time', 'x_wrist', 'y_wrist','Cost']].rename(columns={'x_wrist':'x', 'y_wrist':'y'}).assign(Joint='Wrist')])
g3 = sns.relplot(data = sns_data, x='x', y='y', col='Cost', style='Joint', height=3, aspect=1, markers=['4', '4'], color='grey', alpha=0.5,
                 col_order=[ 'Minimize jerk', 'Minimize jerk and time',   'Minimize jerk, energy and time',], legend=False)
data_smooth = data_smooth.loc[
    (data_smooth.Time ==0) |
    (data_smooth.Time ==0.099) |
    (data_smooth.Time ==0.201) |
    (data_smooth.Time ==0.3)]

colors_cost = ['#1f77b4', '#ff7f0e', '#2ca02c',]
data_smooth.Time = np.round(data_smooth.Time, 1)
for i, icost in enumerate([ 'Minimize jerk', 'Minimize jerk and time',   'Minimize jerk, energy and time',]):
    ax = g3.axes.flatten()[i]
    data = data_smooth[data_smooth.Cost==icost]
    for iirow, irow in data.iterrows():
        ax.plot([0, irow.x_elbow, irow.x_wrist], [0, irow.y_elbow, irow.y_wrist], color=colors_cost[i], alpha = 0.4 + iirow*0.2/34 )
    if icost == 'Minimize jerk, energy and time':
        ax.set_title('Minimize jerk, energy\nand time')
    else:
        ax.set_title(f'{icost}')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(-0.1, 0.8)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.spines[['right', 'top']].set_visible(True)

plt.savefig('./Paper/2DOF_free_in_time.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)



