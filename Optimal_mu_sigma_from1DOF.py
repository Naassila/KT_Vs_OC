# from casadi import *
import numpy as np
from scipy import integrate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from Arm_1DOF import main
from Tools import inflexion_points_logparam_robust_t0_fixed, lognpdf, snr

params = {'legend.fontsize': 'medium',
          'axes.labelsize': 'medium',
          'axes.titlesize':'medium',
          'xtick.labelsize':'medium',
          'ytick.labelsize':'medium'}

plt.rcParams.update(params)
plt.rcParams['font.serif'] ="Times New Roman"
plt.rcParams['font.family'] ="serif"
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams.update({'font.size': 10})
sns.set_context("paper", font_scale=1.3)
ff = mticker.ScalarFormatter(useOffset=False, useMathText=True)
ff.set_powerlimits((-5,5))

N = 28
fsol = np.zeros((1,N))
rs = np.linspace(0.3,0.9,N)
alpha = 1e5
beta = 1


mu_sigma = []
sols = pd.DataFrame(columns=['Movement time', 'α/β', 'mu', 'sigma', 'D', 't0', 'SNR'], data=np.zeros((3*len(rs), 7)))
sols['α/β']= [str(i) for i in sols['α/β']]
q_sols = pd.DataFrame(columns=['Time', 'α/β', 'MT', 'qd', 'qdd'], data=np.zeros((123*len(rs)*4, 5)))
the_sol = []
sol_costs = pd.DataFrame(columns=['Movement time', 'α/β', 'cost_v', 'cost_j',], data=np.zeros((3*len(rs), 4)))
for ia, alpha in enumerate([1000, 1e4, 1e5, 1e6]):
    for ir, r in enumerate(rs):
        isol, all_sol, sol_cost = main(n_shooting=120,
                    final_time=r,
                    cost_function_index=[2],
                    alpha_beta=alpha,
                    inside_plot=False,
                    external_use=True)
        sol_costs.loc[len(rs)*ia+ir] = [r, alpha, np.linalg.norm(sol_cost[:120]), np.linalg.norm(sol_cost[120:])]
        qd = isol['Elbow_vel'].values
        qdd = isol['Elbow_acc'].values
        t=isol.Time.values
        param_log = inflexion_points_logparam_robust_t0_fixed(t, qd, np.gradient(qd, t),
                                                     [0.1, 0.6, 0.1], t0=0.01-0.01, plot_check=False)
        log_for_cost = param_log[2][0] * lognpdf(t, param_log[0][0], param_log[1][0],
                                                     t0=param_log[3][0])
        snr_value = snr(qd, log_for_cost)


        sols.loc[len(rs)*ia+ir] = [r, alpha, param_log[0][0], param_log[1][0], param_log[2][0], param_log[3][0], snr_value]
        q_sols[123*ir+123*len(rs)*ia:123*(ir+1)+123*len(rs)*ia] = np.vstack((t, 123*[alpha], 123*[r], qd, qdd)).T
        if np.round(r, 2)==0.3 and alpha == 5000:
            the_sol=all_sol[0]
if len(set(sols.t0))==1:
    sols= sols.drop(['t0'], axis=1)
data=sols.melt(['Movement time','α/β'])
# data['α/β'] = ['{:.0e}'.format(i) for i in data['α/β']]
data['α/β' ] = ['${}$'.format(ff.format_data(i)) for i in data['α/β'] ]
point_int = data[np.round(data['Movement time'], 2)==0.3]
mu_sigma_opt = point_int[point_int['α/β']==5000].value.values
g = sns.relplot(data=data, x='Movement time', y='value', col='variable',
                col_order = ['mu', 'sigma', 'D',  'SNR'], col_wrap= 2, palette='rocket',
                style='α/β' , hue='α/β', kind='scatter', facet_kws={'sharey': False, 'sharex': True},
                height=2, aspect=2)
ylabels = ['μ', 'σ', 'D', 'SNR[dB]']
ybounds = [[-2.2, -1.0], [0.5, 0.61], [0.4, 1.0], [8, 25]]
for i, ax in enumerate(g.axes.flatten()):
    # if i == 0:
    #     # ax.set_ylim(0, 1)
    #     # ax.set_yticks([0, 0.5, 1])
    #     ax.axhline(-2.2, color='k', linestyle=':', alpha=0.5)
    #     ax.axhline(-1.5, color='k', linestyle=':', alpha=0.5)
    #     # ax.scatter(0.3, mu_sigma_opt[0],s=30, facecolors='none', edgecolors='k' ,zorder=10)
    # elif i == 1:
    #     # ax.set_ylim(-1.8, 0)
    #     # ax.set_yticks([-1.8, -0.9, 0])
    #     ax.axhline(0.6, color='k', linestyle=':', alpha=0.5)
    #     ax.axhline(0.1, color='k', linestyle=':', alpha=0.5)
    #     # ax.scatter(0.3, mu_sigma_opt[1],s=30, facecolors='none', edgecolors='k', zorder=10)
    ax.set_ylabel(ylabels[i])
    ax.set_ylim(ybounds[i])
    ax.set_title('')
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0.3, 0.9)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9])
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.ticklabel_format(style='sci', axis='y')
    ax.spines[['right', 'top']].set_visible(True)

plt.savefig('mu_sigma_opt.svg')
plt.close()

t_03 = np.linspace(0, point_int['Movement time'].values[0], 100)
Nt = t_03[1]
t03_t0 = t_03-mu_sigma_opt[3]
v_03 = -mu_sigma_opt[2] * np.exp(-0.5*(
        (np.log(t03_t0) - mu_sigma_opt[0])/mu_sigma_opt[1]
        )**2) / (t03_t0 * np.sqrt(2*np.pi)* mu_sigma_opt[1])

# v_03[0] = 0
a_03 = np.gradient(v_03, t_03)
j_03 = np.gradient(a_03, t_03)
d_03 = integrate.cumulative_trapezoid(v_03, t_03, initial=0)
d_03 = d_03/d_03.max()

data_03 = pd.concat([pd.DataFrame({'Time [s]': t_03,
                     'Normalized\ndisplacement':d_03,
                     'Velocity [m.s⁻¹]':v_03,
                     'Acceleration [m.s⁻²]': a_03,
                     'Jerk [m.s⁻³]': j_03}).assign(origin='Lognormal profile'),
                     pd.DataFrame({'Time [s]': the_sol.Time,
                     'Normalized\ndisplacement':(the_sol.q1-the_sol.q1[0])/(-100),
                     'Velocity [m.s⁻¹]':the_sol.dq1*0.3,
                     'Acceleration [m.s⁻²]': the_sol.ddq1*0.3,
                     'Jerk [m.s⁻³]': the_sol.dddq1*0.3}).assign(origin='$C_{OCP}$'),
                     ]).melt(['Time [s]', 'origin'])

g = sns.relplot(data=data_03, x='Time [s]', y='value', col='variable', col_wrap=2,
                hue = 'origin',
                col_order= ['Normalized\ndisplacement',
                            'Acceleration [m.s⁻²]',
                            'Velocity [m.s⁻¹]',
                            'Jerk [m.s⁻³]'], kind='line',
                facet_kws={'sharey': False, 'sharex': True},
                height=2, aspect=2)

for i, ax in enumerate(g.axes.flatten()):
    # if i == 0:
    #     # ax.set_ylim(0, 1)
    #     # ax.set_yticks([0, 0.5, 1])
    #     ax.set_ylabel('μ')
    #     ax.axhline(-2.2, color='k', linestyle=':', alpha=0.5)
    #     ax.axhline(-1.5, color='k', linestyle=':', alpha=0.5)
    #     ax.plot(0.3, mu_sigma_opt[0],'ko')
    # else:
    #     # ax.set_ylim(-1.8, 0)
    #     # ax.set_yticks([-1.8, -0.9, 0])
    #     ax.set_ylabel('σ')
    #     ax.axhline(0.6, color='k', linestyle=':', alpha=0.5)
    #     ax.axhline(0.1, color='k', linestyle=':', alpha=0.5)
    #     ax.plot(0.3, mu_sigma_opt[1],'ko')
    ax.set_ylabel(ax.get_title().split('=')[-1])
    ax.set_title('')
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0, 0.3)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    # ax.ticklabel_format(style='sci', axis='y')
    ax.spines[['right', 'top']].set_visible(True)

# plt.subplots_adjust(hspace=0)
plt.savefig('mu_sigma_opt_kin.svg')
plt.close()