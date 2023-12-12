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
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

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


titles = ['qd', 'qdd',]
data = pd.read_csv('discussion.csv')

df = data.melt(['Time', 'α/β', 'MT'])
df = data.drop('qdd', axis=1)

g = sns.relplot(data=df, x='Time', y='qd', col='α/β', col_wrap=2, hue='MT',
                kind = 'line', facet_kws = {'sharey': True, 'sharex': True},
                palette="crest", legend=False,
                height = 2, aspect = 2
                )

for i, ax in enumerate(g.axes.flatten()):
    if i in [0, 2]:
        ax.set_ylabel('Velocity [m.s⁻¹]')
    ax.set_title(ax.get_title()[:-2])
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0, 0.9)
    ax.set_xticks([0, 0.3, 0.6, 0.9])
    ax.set_ylim(0)
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    # ax.ticklabel_format(style='sci', axis='y')
    ax.spines[['right', 'top']].set_visible(True)

norm = plt.Normalize(df.MT.min(), df.MT.max())
sm = plt.cm.ScalarMappable(cmap="crest", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=g.axes[:], shrink=0.5)
cbar.ax.set_title('Movement\nduration')
cbar.set_ticks([0.2,0.55,0.9])

plt.savefig('discussion_mu_sigma_opt_kin.svg')
plt.close()

print('Ottokke !!')