from casadi import *
import numpy as np
from scipy import integrate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

def lognpdf(T, mu, sigma):
    y = exp(-0.5*((log(T) - mu)/sigma)**2) / (T * sqrt(2*np.pi) * sigma)
    return y
def objective_function(mu, sigma, alpha, beta):
    T = linspace(0, exp(mu+3*sigma), 100)
    v = lognpdf(T, mu, sigma)
    dt = T[1]
    a = []
    j = []
    for i in range(v.shape[0]-1):
        a.append((v[i + 1] - v[i]) / dt)
    for i in range(len(a)-1):
        j.append((a[i + 1] - a[i]) / dt)
    cost = (alpha * v[:-2]**2 + beta*vertcat(*j)**2)*T[:-2]**2
    sum_cost = 0
    for i in range(cost.shape[0]-1):
        sum_cost = sum_cost + (cost[i+1]+cost[i])/2*dt

    return sum_cost

N = 28
fsol = np.zeros((1,N))
rs = np.linspace(0.2,0.9,N)
alpha = 10000
beta = 1


def optimize(r, alpha):
    mu = SX.sym('mu')
    sigma = SX.sym('sigma')

    ms = vertcat(mu, sigma)
    f = objective_function(mu, sigma, alpha, beta)
    g = exp(mu + 3 * sigma) - exp(mu - 3 * sigma)
    nlp = {'x': ms, 'f': f, 'g': g}
    solver = nlpsol('solver', 'ipopt', nlp)
    res = solver(x0 = [-2, 0.5],
                 lbx = [-3, 0.1],
                 ubx = [2.0, 0.9],
                 lbg = r,
                 ubg = r)
    return res

mu_sigma = []
sols = pd.DataFrame(columns=['Movement time', 'α/β', 'mu', 'sigma'], data=np.zeros((3*len(rs), 4)))
sols['α/β']= [str(i) for i in sols['α/β']]
for ia, alpha in enumerate([100, 1000, 5000, 1e4]):
    for ir, r in enumerate(rs):
        fsol = optimize(r, alpha)
        sols.loc[len(rs)*ia+ir] = [r, alpha, float(fsol['x'][0]), float(fsol['x'][1])]

data=sols.melt(['Movement time','α/β'])
point_int = data[np.round(data['Movement time'], 2)==0.3]
mu_sigma_opt = point_int[point_int['α/β']==5000].value.values
g = sns.relplot(data=data, x='Movement time', y='value', row='variable',
                hue='α/β', kind='line', facet_kws={'sharey': False, 'sharex': True},
                height=2, aspect=2)

for i, ax in enumerate(g.axes.flatten()):
    if i == 0:
        # ax.set_ylim(0, 1)
        # ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel('μ')
        ax.axhline(-2.2, color='k', linestyle=':', alpha=0.5)
        ax.axhline(-1.5, color='k', linestyle=':', alpha=0.5)
        ax.scatter(0.3, mu_sigma_opt[0],s=30, facecolors='none', edgecolors='k' ,zorder=10)
    else:
        # ax.set_ylim(-1.8, 0)
        # ax.set_yticks([-1.8, -0.9, 0])
        ax.set_ylabel('σ')
        ax.axhline(0.6, color='k', linestyle=':', alpha=0.5)
        ax.axhline(0.1, color='k', linestyle=':', alpha=0.5)
        ax.scatter(0.3, mu_sigma_opt[1],s=30, facecolors='none', edgecolors='k', zorder=10)
    ax.set_title('')
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0.2, 0.9)
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9])
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.ticklabel_format(style='sci', axis='y')
    ax.spines[['right', 'top']].set_visible(True)

plt.savefig('mu_sigma_opt.svg')
plt.close()

t_03 = np.linspace(0, point_int['Movement time'].values[0], 100)
Nt = t_03[1]
v_03 = np.exp(-0.5*((np.log(t_03) - mu_sigma_opt[0])/mu_sigma_opt[1])**2) / (t_03 * np.sqrt(2*np.pi) * mu_sigma_opt[1])
v_03[0] = 0
a_03 = np.gradient(v_03, t_03)
j_03 = np.gradient(a_03, t_03)
d_03 = integrate.cumulative_trapezoid(v_03, t_03, initial=0)
d_03 = d_03/d_03.max()

data_03 = pd.DataFrame({'Time [s]': t_03,
                     'Normalized\ndisplacement':d_03,
                     'Velocity [m.s⁻¹]':v_03,
                     'Acceleration [m.s⁻²]': a_03,
                     'Jerk [m.s⁻³]': j_03}).melt('Time [s]')

g = sns.relplot(data=data_03, x='Time [s]', y='value', col='variable', col_wrap=2,
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


print('yay')




