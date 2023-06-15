import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import itertools
from scipy.special import erfinv
from scipy.integrate import simpson
from scipy.stats import lognorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

from scipy.interpolate import griddata

def multi_plot(fig, df, pos,  z,):
    x = df.mu
    y = df.sigma
    grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]
    grid_z = griddata((x, y), df[z], (grid_x, grid_y), method='cubic')

    ax = fig.add_subplot(2, 3, pos, projection='3d')
    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    ax.set_zlabel('Cost')
    ax.set_title(z)
    ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.viridis)
    ax.set(xlim=[-2.2, -1.5], ylim=[0.1, 0.6])
    ax.set_zscale('log')
    my_cmap = plt.get_cmap('hsv')

#     ax.scatter3D(x, y, df[z],alpha = 0.8, c = df[z], cmap = my_cmap, s=1)\n",
    ax.view_init( 50, -140)
    ax.invert_xaxis()
    ax.invert_yaxis()

def lognpdf(x, mu, sigma):
    x = np.where(x <= 0, np.inf, x )
    y = np.exp(-0.5*((np.log(x) - mu)/sigma)**2) / (x * np.sqrt(2*np.pi) * sigma)
    return y

def a_using_norm(x, t, T, Nt):
    v_int = lognpdf(t, x['mu'], x['sigma'])
    a_int = np.diff(v_int)/(T/Nt)
    return np.linalg.norm(a_int, ord=2)

def j_using_norm(x, t, T, Nt):
    v_int = lognpdf(t, x['mu'], x['sigma'])
    a_int = np.diff(v_int)/(T/Nt)
    j_int = np.diff(a_int) / (T / Nt)
    return np.linalg.norm(j_int, ord=2)

def a_using_simps(x, t, T, Nt):
    v_int = lognpdf(t, x['mu'], x['sigma'])
    a_int = np.diff(v_int)/(T/Nt)
    return np.sqrt(simpson(a_int ** 2, x=t[:-1]))

def j_using_simps(x, t, T, Nt):
    v_int = lognpdf(t, x['mu'], x['sigma'])
    a_int = np.diff(v_int)/(T/Nt)
    j_int = np.diff(a_int) / (T / Nt)
    return np.sqrt(simpson(j_int ** 2, x=t[:-2]))

def kinetic_time_using_simps(x, t, T, Nt):
    t_from_MT = np.linspace(0, x['MT'], Nt)
    v_int = lognpdf(t_from_MT, x['mu'], x['sigma'])
    vt = v_int*t_from_MT
    return np.sqrt(simpson(vt**2, x=t_from_MT, axis=0))

def jerk_time_using_simps(x, t, T, Nt):
    t_from_MT = np.linspace(0, x['MT'], Nt)
    v_int = lognpdf(t_from_MT, x['mu'], x['sigma'])
    a_int = np.diff(v_int)/(x['MT']/Nt)
    j_int = np.diff(a_int) / (x['MT'] / Nt)
    jt = j_int*t_from_MT[:-2]
    return np.sqrt(simpson(jt**2, x=t_from_MT[:-2]))


nb = 11
NB = 21

Nt = 1000
T = 0.7
t = np.linspace(0, T, Nt)

# Lognormal definitions
t0_vect = np.linspace(0.05, 1, NB)
D1_vect = np.linspace(0.5, 7, NB)
D2_vect = np.linspace(5, 70, NB)
mu_vect = np.linspace(-3.0, -1.5, NB) #-- -1.5 in paper, -2.2 before
sigma_vect = np.linspace(0.1, 0.6, NB) #---0.6 in paper

t0 = 0
D = 1
sigma = 0.25
mu = -2

# Movement time as defined from a lognormal
MT = np.zeros((NB, NB))
for n in range(NB):
    for l in range(NB):
        MT[n, l] = 2*np.exp(mu_vect[n])*np.sinh(2*sigma_vect[l])

#Using the quantile function allows us to determine precisely for each
#value of (mu, sigma), the time to reach any percentile
MT_quant = np.zeros((NB, NB))
for n in range(NB):
    for l in range(NB):
        MT_quant[n, l] = np.exp(mu_vect[n]+np.sqrt(2*sigma_vect[l]**2)*erfinv(2*0.997-1))

# Movement time as the lognormal variance
MT_variance = np.zeros((NB, NB))
for n in range(NB):
    for l in range(NB):
        MT_variance[n, l] = np.exp(2*mu_vect[n]+2*sigma_vect[l]**2) - np.exp(2*mu_vect[n]+sigma_vect[l]**2)

k = np.zeros(len(t))
for i in range(len(t)):
    k[i] = (np.log(t[i]-t0)-mu)/sigma

v = lognpdf(t-t0, mu, sigma)
vlog = np.log(v)
a = np.diff(v)/(T/Nt)
j = np.diff(a)/(T/Nt)

plt.figure()
plt.plot(t, v, label ='velocity')
plt.plot(t[:-1], a/100, label='acceleration/100')
plt.plot(t[:-2], j/1000, label ='Jerk/1000')
plt.legend()
plt.show(block=False)
plt.close()

v_L2 = np.zeros((NB, NB))
a_L2 = np.zeros((NB, NB))
j_L2 = np.zeros((NB, NB))

v_simps = np.zeros((NB, NB))
a_simps = np.zeros((NB, NB))
j_simps = np.zeros((NB, NB))

plot_timeseries_mu_sigma = False
if plot_timeseries_mu_sigma:
     # Mu variable, sigma set
    sigma = 0.35 #Sigma is set to its median value to avoid fringe effects
    v_vect_mu = np.zeros((Nt, NB))
    a_vect_mu = np.zeros((Nt-1, NB))
    j_vect_mu = np.zeros((Nt-2, NB))
    mu_var = pd.DataFrame({'time':t})

    for n in range(NB):
        mu = mu_vect[n]
        v_vect_mu[:, n] = lognpdf(t-t0, mu, sigma)
        a_vect_mu[:, n] = np.diff(v_vect_mu[:, n])/(T/Nt)
        j_vect_mu[:, n] = np.diff(a_vect_mu[:, n])/(T/Nt)
        col_name = f'v_mu_{mu}_sigma_{sigma}'
        mu_var = mu_var.join(pd.DataFrame({col_name:v_vect_mu[:, n]}))
        col_name = f'a_mu_{mu}_sigma_{sigma}'
        mu_var = mu_var.join(pd.DataFrame({col_name: a_vect_mu[:, n]}))
        col_name = f'j_mu_{mu}_sigma_{sigma}'
        mu_var = mu_var.join(pd.DataFrame({col_name: j_vect_mu[:, n]}))

    # Sigma variable, mu set
    mu = -1.9 #Sigma is set to its median value to avoid fringe effects
    v_vect_sigma = np.zeros((Nt, NB))
    a_vect_sigma = np.zeros((Nt-1, NB))
    j_vect_sigma = np.zeros((Nt-2, NB))
    sigma_var = pd.DataFrame({'time':t})
    for n in range(NB):
        sigma = sigma_vect[n]
        v_vect_sigma[:, n] = lognpdf(t-t0, mu, sigma)
        a_vect_sigma[:, n] = np.diff(v_vect_sigma[:, n])/(T/Nt)
        j_vect_sigma[:, n] = np.diff(a_vect_sigma[:, n])/(T/Nt)
        col_name = f'v_mu_{mu}_sigma_{sigma}'
        sigma_var = sigma_var.join(pd.DataFrame({col_name:v_vect_sigma[:, n]}))
        col_name = f'a_mu_{mu}_sigma_{sigma}'
        sigma_var = sigma_var.join(pd.DataFrame({col_name: a_vect_sigma[:, n]}))
        col_name = f'j_mu_{mu}_sigma_{sigma}'
        sigma_var = sigma_var.join(pd.DataFrame({col_name: j_vect_sigma[:, n]}))


    sigma_var = sigma_var.assign(which_var='sigma')
    mu_var = mu_var.assign(which_var='mu')
    df_sigma_mu = pd.concat([sigma_var, mu_var])
    df_sigma_mu = df_sigma_mu.melt(id_vars=['time', 'which_var'])
    df_sigma_mu['var_type'] = [i.split('_')[0] for i in df_sigma_mu.variable]
    df_sigma_mu['sigma'] = [float(i.split('_')[-1]) for i in df_sigma_mu.variable]
    df_sigma_mu['mu'] = [float(i.split('_')[2]) for i in df_sigma_mu.variable]
    condition = [df_sigma_mu.which_var.eq('sigma'), df_sigma_mu.which_var.eq('mu')]
    choice = [df_sigma_mu.sigma, df_sigma_mu.mu]
    df_sigma_mu['sigma_mu'] = np.select(condition, choice)

    # Plot timeseries response as a function of sigma ------------------------------------
    g_sigma = sns.relplot(data=df_sigma_mu[df_sigma_mu.which_var=='sigma'], x='time', y='value', hue='sigma_mu', row='var_type', col='which_var',
                    kind='line', facet_kws={'sharey': False, 'sharex': True}, legend=False, palette='winter', aspect=2, height=2.5)
    norm = plt.Normalize(df_sigma_mu.sigma.min(), df_sigma_mu.sigma.max())
    sm = plt.cm.ScalarMappable(cmap="winter", norm=norm)
    sm.set_array([])
    var_list = ['Velocity [m.s⁻¹]', 'Acceleration [m.s⁻²]', 'Jerk [m.s⁻³]']
    cbar = plt.colorbar(sm, ax=g_sigma.axes[:], shrink=0.5)
    cbar.ax.set_title('σ')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    for i, ax in enumerate(g_sigma.axes.flatten()):
        if i == 0:
            ax.set_title('Profiles for μ = -1.9', y=1.2)
        else:
            ax.set_title('')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(var_list[i])
        ax.set_xlim(0, 0.7)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.ticklabel_format(style='sci', axis='y')
        ax.yaxis.set_major_formatter(formatter)
        ax.spines[['right', 'top']].set_visible(True)
    plt.savefig('VAJ_time_sigma.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close()

    # Plot timeseries response as a function of mu ------------------------------------
    g_mu = sns.relplot(data=df_sigma_mu[df_sigma_mu.which_var=='mu'], x='time', y='value', hue='sigma_mu', row='var_type', col='which_var',
                    kind='line', facet_kws={'sharey': False, 'sharex': True}, legend=False, palette='cool', aspect=2, height=2.5)
    norm = plt.Normalize(df_sigma_mu.mu.min(), df_sigma_mu.mu.max())
    sm = plt.cm.ScalarMappable(cmap="cool", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=g_mu.axes[:], shrink=0.5)
    cbar.ax.set_title('μ')
    for i, ax in enumerate(g_mu.axes.flatten()):
        if i == 0:
            ax.set_title('Profiles for σ = 0.35', y=1.2)
        else:
            ax.set_title('')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(var_list[i])
        ax.set_xlim(0, 0.7)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.ticklabel_format(style='sci', axis='y')
        ax.yaxis.set_major_formatter(formatter)
        ax.spines[['right', 'top']].set_visible(True)
    plt.savefig('VAJ_time_mu.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close()

print('so far so good')

cost = pd.DataFrame(columns=['mu', 'sigma', 'MT',
                             'v', 'a', 'j', 'A) Kinetic Energy',
                             'B) Jerk', 'C) Jerk & Kinetic Energy','D) Kinetic Energy x Time',
                             'E) Jerk x Time', 'F) (Jerk & Kinetic Energy) x Time'],
                    index=np.arange(1, sigma_vect.shape[0]*mu_vect.shape[0]+1, 1))
cost_BEN = pd.DataFrame(columns=['mu', 'sigma', 'MT',
                                 'v', 'a', 'j', 'A) Kinetic Energy',
                                 'B) Jerk', 'C) Jerk & Kinetic Energy','D) Kinetic Energy x Time',
                                 'E) Jerk x Time', 'F) (Jerk & Kinetic Energy) x Time'],
                        index=np.arange(1, sigma_vect.shape[0]*mu_vect.shape[0]+1, 1))
mu_sigma = np.array(list(itertools.product(mu_vect, sigma_vect)))
cost.mu = mu_sigma.T[0]
cost.sigma = mu_sigma.T[1]
cost.MT = 2*np.exp(cost.mu)*np.sinh(3*cost.sigma)

cost_BEN.mu = mu_sigma.T[0]
cost_BEN.sigma = mu_sigma.T[1]
cost_BEN.MT = 2*np.exp(cost.mu)*np.sinh(3*cost.sigma)

cost.v = cost.apply(lambda x: np.sqrt(simpson(lognpdf(t, x['mu'], x['sigma'])**2, x=t, axis=0)), axis=1)
cost.a = cost.apply(lambda x: a_using_simps(x, t, T, Nt), axis=1)
cost.j = cost.apply(lambda x: j_using_simps(x, t, T, Nt), axis=1)

cost_BEN.v = cost_BEN.apply(lambda x: np.linalg.norm(lognpdf(t, x['mu'], x['sigma']), ord=2), axis=1)
cost_BEN.a = cost_BEN.apply(lambda x: a_using_norm(x, t, T, Nt), axis=1)
cost_BEN.j = cost_BEN.apply(lambda x: j_using_norm(x, t, T, Nt), axis=1)

cost['A) Kinetic Energy'] = cost.v
cost['B) Jerk'] = cost.j
cost['C) Jerk & Kinetic Energy'] = cost.j + 1000*cost.v
cost['D) Kinetic Energy x Time'] = cost.MT*cost.v#cost.apply(lambda x: kinetic_time_using_simps(x, t, T, Nt), axis=1)
cost['E) Jerk x Time'] = cost.MT*cost.j#cost.apply(lambda x: jerk_time_using_simps(x, t, T, Nt), axis=1)
cost['F) (Jerk & Kinetic Energy) x Time'] = cost['E) Jerk x Time'] + 1000 * cost['D) Kinetic Energy x Time']

cost_BEN['A) Kinetic Energy'] = cost_BEN.v
cost_BEN['B) Jerk'] = cost_BEN.j
cost_BEN['C) Jerk & Kinetic Energy'] = cost_BEN.j + 1000*cost_BEN.v
cost_BEN['D) Kinetic Energy x Time'] = cost_BEN.MT*cost_BEN.v
cost_BEN['E) Jerk x Time'] = cost_BEN.MT*cost_BEN.j
cost_BEN['F) (Jerk & Kinetic Energy) x Time'] = cost_BEN.MT*cost_BEN['C) Jerk & Kinetic Energy']

plot3d = False
if plot3d:
    titles = ['A) Kinetic Energy', 'B) Jerk', 'C) Jerk & Kinetic Energy',
                                'D) Kinetic Energy x Time', 'E) Jerk x Time', 'F) (Jerk & Kinetic Energy) x Time']
    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes=True,
                        specs = [[{'type':'surface'}, {'type':'surface'}, {'type':'surface'}],
                                 [{'type':'surface'}, {'type':'surface'}, {'type':'surface'}]],
                        subplot_titles = titles, x_title='', y_title= '', horizontal_spacing = 0.05, vertical_spacing=0.05)

    for ii, iplot in enumerate(titles):
        row = ii//3
        col = ii%3
        df = cost[['mu', 'sigma', iplot]]
        df = df.pivot(values=iplot, columns='mu', index='sigma')
        fig.add_trace(go.Surface(x=df.columns.values, y=df.index.values, z=df.values,  colorscale='Viridis',
                                 # surfacecolor=2*np.exp(df.columns.values)*np.sinh(df.index.values), cmin = cost.MT.min(), cmax=cost.MT.max(),
                                 contours=  {"z": {"show": True, "start": df.values.min(), "end": df.values.max(), "size": (df.values.max()-df.values.min())/20,
                                                   "color":"white"}},
                                 showscale=False,), row=row+1, col=col+1,
                     )
    fig.update_layout(
        font_family="Times New Roman",
        title_font_family="Times New Roman",
    )
    fig.update_xaxes(title_font_family="Times New Roman")
    fig.update_yaxes(title_font_family="Times New Roman")
    fig.update_scenes(xaxis_title_text='μ',
                      yaxis_title_text='σ',
                      zaxis_title_text='')
    fig.write_html("cost_functions.html")
    fig.show()

df_mu_19 = cost[cost.mu==-1.85]
df = df_mu_19[['sigma','v', 'a', 'j']].melt('sigma')
g_mu = sns.relplot(data=df, x='sigma', y='value', row='variable', kind='line',
            facet_kws={'sharey': False, 'sharex': True}, legend=False, aspect=2, height=2.5)
col_names = ['Velocity norm', 'Acceleration norm', 'Jerk norm']
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
for i, ax in enumerate(g_mu.axes.flatten()):
    if i == 0:
        ax.set_title('Profiles for μ = -1.85', y=1.2)
    else:
        ax.set_title('')
    ax.set_xlabel('σ')
    ax.set_ylabel(col_names[i])
    ax.set_xlim(0.1, 0.6)
    ax.set_xticks([0.1, 0.35, 0.6])
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.ticklabel_format(style='sci', axis='y')
    ax.yaxis.set_major_formatter(formatter)
    ax.spines[['right', 'top']].set_visible(True)
plt.savefig('VAJ_cost_sigma.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.close()

df_sigma_35 = cost[cost.sigma==0.35]
df = df_sigma_35[['mu','v', 'a', 'j']].melt('mu')
g_sigma = sns.relplot(data=df, x='mu', y='value', row='variable', kind='line',
            facet_kws={'sharey': False, 'sharex': True}, legend=False, aspect=2, height=2.5)
for i, ax in enumerate(g_sigma.axes.flatten()):
    if i == 0:
        ax.set_title('Profiles for σ = 0.35', y=1.2)
    else:
        ax.set_title('')
    ax.set_xlabel('μ')
    ax.set_ylabel(col_names[i])
    ax.set_xlim(-2.2, -1.5)
    ax.set_xticks([-2.2, -1.85, -1.5])
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.ticklabel_format(style='sci', axis='y')
    ax.yaxis.set_major_formatter(formatter)
    ax.spines[['right', 'top']].set_visible(True)
plt.savefig('VAJ_cost_mu.svg', dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.close()


fig_B = plt.figure(figsize=(14, 6), dpi = 600) #plt.figaspect(0.7))
fig_B.suptitle('From BEN')
multi_plot(fig_B, cost_BEN, 1, 'A) Kinetic Energy')
multi_plot(fig_B, cost_BEN, 2, 'B) Jerk', )
multi_plot(fig_B, cost_BEN, 3, 'C) Jerk & Kinetic Energy',)
multi_plot(fig_B, cost_BEN, 4, 'D) Kinetic Energy x Time', )
multi_plot(fig_B, cost_BEN, 5, 'E) Jerk x Time', )
multi_plot(fig_B, cost_BEN, 6, 'F) (Jerk & Kinetic Energy) x Time')
plt.tight_layout()

fig = plt.figure(figsize=(14, 6), dpi = 600) #plt.figaspect(0.7))
fig.suptitle('Using simps')
multi_plot(fig, cost, 1, 'A) Kinetic Energy')
multi_plot(fig, cost, 2, 'B) Jerk', )
multi_plot(fig, cost, 3, 'C) Jerk & Kinetic Energy',)
multi_plot(fig, cost, 4, 'D) Kinetic Energy x Time', )
multi_plot(fig, cost, 5, 'E) Jerk x Time', )
multi_plot(fig, cost, 6, 'F) (Jerk & Kinetic Energy) x Time')
plt.tight_layout()

print('Ottokke !!')