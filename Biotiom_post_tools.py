import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

def plot_kinematics(t, q, qd, qdd, jerk, name):
    fig1, ax1 = plt.subplots(nrows=1, ncols=4)
    ax1[0].plot(t, np.rad2deg(q[0]), label="q0")
    ax1[0].set_title('Angle')
    ax1[1].plot(t,qd[0], label="q0")
    ax1[1].set_title('Velocity')
    ax1[2].plot(t,qdd[0], label="q0")
    ax1[2].set_title('Acceleration')
    ax1[3].plot(t,jerk[0], label="q0")
    ax1[3].set_title('Jerk')

    try:
        ax1[0].plot(t, np.rad2deg(q[1]), label="q1")
        ax1[1].plot(t, qd[1], label="q1")
        ax1[2].plot(t, qdd[1], label="q1")
        ax1[3].plot(t, jerk[1], label="q1")
    except:
        print('problem with 1DOF')
    plt.legend()
    plt.savefig(name)
    plt.close()

def plot_interarticular(data, var_list, name):
    fig, ax = plt.subplots(figsize=(6, 6))
    g = sns.lineplot(data=data, x='q2', y='q1', hue='cost', sort=False, ax=ax)
    g.set_ylabel(var_list[0])
    g.set_xlabel(var_list[1])
    plt.legend(frameon=False)
    g.axes.set_aspect('equal', adjustable='box')
    plt.savefig(name)
    plt.close()

def plot_q_qdot_prx_dist(data, var_list, name):
    final_time = data.Time.max()
    df = data.melt(['Time', 'cost'])
    df['Type'] = [i[0] for i in df.variable]
    df['Joint'] = [i[-2:] for i in df.variable]

    g = sns.relplot(data=df, x='Time', y='value', hue='cost', row='Type', col='Joint',
                    col_order = ['q1', 'q2'], kind='line',
                    facet_kws={'sharey': False, 'sharex': True},
                    height=2, aspect=1.5
                    )
    # g.axes[0][0].set_ylim(-70, 30)
    # g.axes[0][0].set_yticks([-70, -20, 30])
    g.axes[0][0].axhline(-70, color='k', linestyle=':', alpha=0.5)
    g.axes[0][0].axhline(30, color='k', linestyle=':', alpha=0.5)
    g.axes[0][0].set_ylabel(var_list[0])
    g.axes[0][0].set_title('Proximal joint')

    # g.axes[0][1].set_ylim(10, 60)
    # g.axes[0][1].set_yticks([10, 35, 60])
    g.axes[0][1].axhline(10, color='k', linestyle=':', alpha=0.5)
    g.axes[0][1].axhline(60, color='k', linestyle=':', alpha=0.5)
    g.axes[0][1].set_title('Distal joint')

    g.axes[1][0].set_ylabel(var_list[1])
    g.axes[1][0].set_title('')

    g.axes[1][1].set_title('')
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_xlabel('Time [s]')
        ax.set_xlim(0, final_time)
        #
        # ax.axvline(0.2, color='k', linestyle=':', alpha=0.5)
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.ticklabel_format(style='sci', axis='y')
        ax.spines[['right', 'top']].set_visible(True)
    plt.savefig(name)
    plt.close()

def plot_effector_velocity(marker_position, name):
    final_time = marker_position.Time.max()
    effectors_velocity = marker_position[['Time', 'Wrist_vel', 'Elbow_vel', 'cost']].melt(['Time', 'cost'])
    g2 = sns.relplot(data=effectors_velocity, x='Time', y='value', col='variable', hue='cost',
                     kind='line', col_order=['Elbow_vel', 'Wrist_vel'], facet_kws={'sharey': False, 'sharex': True},
                     height=2, aspect=1.3)
    g2.axes[0][0].set_ylabel('Velocity [m/s]')
    for i, ax in enumerate(g2.axes.flatten()):
        ax.set_title(ax.get_title().split()[-1][:-4])
        ax.set_xlabel('Time [s]')
        ax.set_xlim(0, final_time)
        # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.ticklabel_format(style='sci', axis='y')
        ax.spines[['right', 'top']].set_visible(True)
    plt.savefig(name)
    plt.close()

def plot_effector_acceleration(marker_position, name):
    final_time = marker_position.Time.max()
    effectors_velocity = marker_position[['Time', 'Wrist_acc', 'Elbow_acc', 'cost']].melt(['Time', 'cost'])
    g2 = sns.relplot(data=effectors_velocity, x='Time', y='value', col='variable', hue='cost',
                     kind='line', col_order=['Elbow_acc', 'Wrist_acc'], facet_kws={'sharey': False, 'sharex': True},
                     height=2, aspect=1.3)
    g2.axes[0][0].set_ylabel('Acceleration [m.s⁻²]')
    for i, ax in enumerate(g2.axes.flatten()):
        ax.set_title(ax.get_title().split()[-1][:-4])
        ax.set_xlabel('Time [s]')
        ax.set_xlim(0, final_time)
        # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.ticklabel_format(style='sci', axis='y')
        ax.spines[['right', 'top']].set_visible(True)
    plt.savefig(name)
    plt.close()

def plot_arm_in_time(marker_position, col_order, name):
    final_time = marker_position.Time.max()
    try:
        data_smooth = pd.concat([
            marker_position[['Time', 'Elbow_x', 'Elbow_y', 'cost' ]].rename(columns={'Elbow_x': 'x', 'Elbow_y': 'y'}).assign(
                Joint='Elbow'),
            marker_position[['Time', 'Wrist_x', 'Wrist_y', 'cost']].rename(columns={'Wrist_x': 'x', 'Wrist_y': 'y'}).assign(
                Joint='Wrist')
        ])
    except:
        data_smooth = pd.concat([
            marker_position[['Time', 'Elbow_x', 'Elbow_y', 'cost']].rename(
                columns={'Elbow_x': 'x', 'Elbow_y': 'y'}).assign(
                Joint='Elbow'),
        ])


    g3 = sns.relplot(data=data_smooth, x='x', y='y', style='Joint', height=3, aspect=1.0, markers=['4', '4'],
                     col='cost',
                     color='grey', alpha=0.5,
                     col_order=col_order,
                     legend=False)
    try:
        data = marker_position[['Time', 'Elbow_x', 'Elbow_y', 'Wrist_x', 'Wrist_y', 'cost']]
        elbow_Only = False
    except:
        data = marker_position[['Time', 'Elbow_x', 'Elbow_y', 'cost']]
        elbow_Only = True
    s_data = pd.DataFrame(columns=data.columns, data=np.zeros((5*3, len(data.columns))))
    time_of_interest = [0, 0.2, 0.4, 0.6, final_time]
    s_data.Time = time_of_interest*3
    for i, icost in enumerate(set(data.cost)):
        interpolation_base = data[data.cost == icost]
        s_data.cost[i*len(time_of_interest):(i+1)*len(time_of_interest)] = [icost]*len(time_of_interest)
        for column in interpolation_base.columns[1:-1]:
            s_data[column][i*len(time_of_interest):(i+1)*len(time_of_interest)]= np.interp(time_of_interest, interpolation_base.Time.values, interpolation_base[[column]].values.flatten())
    # data.Time = data.Time.round(3)
    # data = data.loc[
    #     (data.Time == 0) |
    #     (data.Time == 0.2) |
    #     (data.Time == 0.4) |
    #     (data.Time == 0.6) |
    #     (data.Time == final_time)]

    colors_cost = ['#1f77b4', '#ff7f0e', '#2ca02c',]
    colors_palette_cost = [sns.color_palette(palette='ocean')[1:], #GnBu
                           sns.color_palette(palette='gist_heat')[1:],
                           sns.color_palette(palette='viridis')[1:]]
    alphas = np.linspace(3, 10, 5)/10
    for i, iax in enumerate(g3.axes.flatten()):
        icost = col_order[i]
        for iirow, irow in s_data[s_data.cost==icost].iterrows():
            if not elbow_Only:
                iax.plot([0, irow.Elbow_x, irow.Wrist_x], [0, irow.Elbow_y, irow.Wrist_y],
                         color = colors_palette_cost[i][iirow%len(time_of_interest)],
                     # color=colors_cost[i],
                     # alpha=alphas[iirow%len(time_of_interest)]
                         label= irow.Time)#alphas[int((iirow-121*i)/40)])
            else:
                iax.plot([0, irow.Elbow_x], [0, irow.Elbow_y],
                         color=colors_palette_cost[i][iirow % len(time_of_interest)]
                         # color=colors_cost[i],
                         # alpha=alphas[iirow%len(time_of_interest)]
                         )#alphas[int((iirow-121*i)/40)])

        iax.set_title('')
        iax.set_xlabel('')
        iax.set_ylabel('')
        iax.set_xlim(-0.1, 0.6)
        iax.set_ylim(-0.6, 0.6)
        iax.set_aspect('equal')
        iax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        iax.tick_params(axis='y', which='both', left=False, labelleft=False)
        iax.spines[['right', 'top']].set_visible(True)
        labelLines(iax.get_lines(), align=False, fontsize=10)
        plt.tight_layout()

    plt.savefig(name)
    plt.close()

from math import atan2,degrees

# from https://stackoverflow.com/questions/16992038/how-to-place-inline-labels-in-a-line-plot
#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    x = xdata[1]*2/3

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')

    #Plot on the first segment
    ip = 1

    # y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])
    y = ydata[1]*2/3

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)

def plot_cost(sol, n_shooting, name):
    if type(n_shooting) is list:
        cost_1 = [i['cost_value'] for i in sol.detailed_cost if i['name']=='Mayer.CUSTOM']
        cost_2 = [i['cost_value'] for i in sol.detailed_cost if i['name']== 'MultinodeObjectiveFcn.CUSTOM']
        cost_1_w = [i['cost_value_weighted'] for i in sol.detailed_cost if i['name']=='Mayer.CUSTOM']
        cost_2_w = [i['cost_value_weighted'] for i in sol.detailed_cost if i['name']== 'MultinodeObjectiveFcn.CUSTOM']
    else:
        if len(sol.detailed_cost) == n_shooting:
            cost_1 = [i['cost_value'] for i in sol.detailed_cost]
            cost_2 = [1] * n_shooting
            cost_1_w = [i['cost_value_weighted'] for i in sol.detailed_cost]
            cost_2_w = [0] * n_shooting
        else:
            cost_1 = [i['cost_value'] for i in sol.detailed_cost[:n_shooting]]
            cost_2 = [i['cost_value'] for i in sol.detailed_cost[n_shooting:]]
            cost_1_w = [i['cost_value_weighted'] for i in sol.detailed_cost[:n_shooting]]
            cost_2_w = [i['cost_value_weighted'] for i in sol.detailed_cost[n_shooting:]]


    fig1, ax1 = plt.subplots(nrows=1, ncols=2)
    ax1[0].plot(cost_1, label='1st cost_function')
    ax1[0].plot(cost_2, label='2nd cost_function')
    ax1[0].set_title('Not weighted')
    ax1[1].plot(cost_1_w, label='1st cost_function')
    ax1[1].plot(cost_2_w, label='2nd cost_function')
    ax1[1].set_title('Weighted')
    fig1.suptitle( f'The 1st cost function had w= {cost_1_w[2]/cost_1[2]}, '
                  f'and the second w={cost_2_w[2]/cost_2[2]}')
    plt.legend()
    plt.savefig(name)
    plt.close()
