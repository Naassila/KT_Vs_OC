import platform
# import bioviz
import pandas as pd
from bioptim import OdeSolverBase
from Biotiom_post_tools import *
from Bioptim_functions_ocp import *
from Tools import inflexion_points_logparam_robust, lognpdf, inflexion_points_logparam_robust_t0_fixed, snr

def prepare_ocp(
    biorbd_model_path: str,
    n_shooting:int,
    final_time:float,
    cost_dt2: bool = True,
    cost_marker: bool = True,
    min_jerk: bool = True,
    min_energy: bool = True,
    alpha_beta: float = 5000,
    t0:float = 0,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    assume_phase_dynamics: bool = False,
) -> OptimalControlProgram:

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamics, expand=True)

    # Add objective functions
    objective_functions = ObjectiveList()
    multinode_objectives = MultinodeObjectiveList()

    if min_jerk and min_energy and cost_dt2 and cost_marker:
        for i in range(n_shooting):
            # end_effector_velocity * t
            objective_functions.add(
                min_marker_velocity_dt2,
                marker="wrist",
                custom_type=ObjectiveFcn.Mayer,
                node=i,
                quadratic=False,
                weight=alpha_beta,
                expand=True,
            )

        for i in range(n_shooting):
            # end_effector_jerk * t
            multinode_objectives.add(
                multinode_min_marker_jerk_dt2,
                marker="wrist",
                nodes_phase=[0, 0],
                nodes=[i, i + 1],
                weight=1,
                quadratic=False,
                expand=True,
            )
    elif min_jerk and not min_energy and cost_dt2 and cost_marker:
        # end_effector_velocity
        # objective_functions.add(
        #     ObjectiveFcn.Lagrange.MINIMIZE_MARKERS_VELOCITY,
        #     marker_index="wrist",
        #     node=Node.ALL,
        #     weight=1000,
        #     quadratic=True,
        #     expand=True,
        #     derivative=False,
        # ),

        # end_effector_jerk
        for i in range(n_shooting):
            # end_effector_jerk * t
            multinode_objectives.add(
                multinode_min_marker_jerk_dt2,
                marker="wrist",
                nodes_phase=[0, 0],
                nodes=[i, i + 1],
                weight=1,
                quadratic=False,
                expand=True,
            )
        # objective_functions.add(
        #     ObjectiveFcn.Lagrange.MINIMIZE_MARKERS_ACCELERATION,
        #     marker_index="wrist",
        #     weight=1,
        #     quadratic=True,
        #     expand=True,
        #     derivative=True,
        # ),
    elif min_jerk and not min_energy and not cost_dt2 and cost_marker:
        # end_effector_jerk
        for i in range(n_shooting):
            # end_effector_jerk
            multinode_objectives.add(
                multinode_min_marker_jerk,
                marker="wrist",
                nodes_phase=[0, 0],
                nodes=[i, i + 1],
                weight=1,
                quadratic=False,
                expand=True,
            )

    elif min_jerk and min_energy and cost_dt2 and not cost_marker:
        for i in range(n_shooting):
            # angular_velocity * t
            objective_functions.add(
                min_qdot_dt2,
                custom_type=ObjectiveFcn.Mayer,
                node=i,
                command_delay = t0,
                quadratic=False,
                weight=alpha_beta,
            )
        for i in range(n_shooting):
            # angular_jerk * t
            objective_functions.add(
                min_jerk_dt2,
                custom_type=ObjectiveFcn.Mayer,
                node=i,
                command_delay=t0,
                quadratic=False,
                weight=1,
            )

    elif min_jerk and not min_energy and cost_dt2 and not cost_marker:
        for i in range(n_shooting):
            # angular_jerk * t
            objective_functions.add(
                min_jerk_dt2,
                custom_type=ObjectiveFcn.Mayer,
                node=i,
                command_delay=t0,
                quadratic=False,
                weight=1,
            )

    elif min_jerk and not min_energy and not cost_dt2 and not cost_marker:
        # # angular_velocity
        # objective_functions.add(
        #     ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        #     key="qdot",
        #     node=Node.ALL,
        #     weight=1000,
        #     quadratic=True,
        #     expand=True,
        #     derivative=False,
        # ),

        # angular_jerk
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qdddot", weight=1, quadratic=True, expand=True, derivative=True
        ),

    # Path constraint
    d2r = np.pi / 180
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model.bounds_from_ranges("q"))

    # start position
    x_bounds["q"][0, 0] = 30 * d2r

    # end position
    x_bounds["q"][0, -1] = -70 * d2r


    x_bounds["qdot"] = [-600*d2r], [600*d2r] #[-30], [30] #updated to velocity 600 deg/s based on pigeon et al. 2003
    x_bounds["qddot"] = [-5000*d2r], [5000*d2r] #[-2000], [2000] #updated to  acceleration 5000 deg/s2 based on pigeon et al. 2003

    x_bounds["qdot"][:, [0, -1]] = 0.0  # qdot = 0 at start and end
    x_bounds["qddot"][:, [0, -1]] = 0.0  # qDdot = 0 at start and end

    # Define control path constraint
    jerk_min, jerk_max = -4000, 4000 #-4e02, 4e02 # updated to jerk at 170000 pigeon et al. 2003 derivated
    u_bounds = BoundsList()
    u_bounds.add("qdddot", min_bound=[jerk_min] * bio_model.nb_tau, max_bound=[jerk_max] * bio_model.nb_tau)
    u_bounds["qdddot"][:, [0,-1]] = 0.0
    # Initial guess (optional since it is 0, we show how to initialize anyway)
    #    x_init = InitialGuessList()
    #    x_init.add("q", min_bound=[10, 140], max_bound=[70, 20], interpolation=InterpolationType.LINEAR)

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        multinode_objectives=multinode_objectives,
        # constraints=constraints,
        # x_init= x_init,
        ode_solver=ode_solver,
        use_sx=use_sx,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main(n_shooting, final_time, cost_function_index = [0,1,2],
         alpha_beta = 5000, t0 = 0.01,
         inside_plot = True, external_use=False, evaluate_SNR=True):
    """
    Runs the optimization and animates it
    """

    model_path = "arm_1DOF.bioMod"

    # Problem parameters

    obj_dict = {'Minimize jerk':{'cost_dt2':False, 'cost_marker':False,
                                 'min_energy': False, 'min_jerk':True},
                'Minimize jerk and time':{'cost_dt2':True, 'cost_marker':False,
                                          'min_energy': False, 'min_jerk':True},
                'Minimize jerk, energy and time':{'cost_dt2':True, 'cost_marker':False,
                                                  'min_energy': True, 'min_jerk':True}, }
    all_sol = []
    all_functions = [iobj for i,iobj in enumerate(list(obj_dict.keys())) if i in cost_function_index]
    for ifunc in all_functions:

        ocp = prepare_ocp(biorbd_model_path=model_path,
                          cost_dt2=obj_dict[ifunc]['cost_dt2'],
                          cost_marker=obj_dict[ifunc]['cost_marker'],
                          min_energy=obj_dict[ifunc]['min_energy'],
                          min_jerk=obj_dict[ifunc]['min_jerk'],
                          n_shooting=n_shooting,
                          final_time=final_time-t0,
                          alpha_beta=alpha_beta,
                          t0=t0)

        # --- Solve the program --- #
        sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
        print(f'Solver for {ifunc} with {final_time}s, and a ratio a/b of {alpha_beta}'
              f' ended with {sol.status}')
        if ifunc == 'Minimize jerk, energy and time':
            plot_cost(sol, n_shooting, f'Q1DOF_bio_v1_costs_{ifunc}.png')
        q = sol.states["q"]
        qd = sol.states["qdot"]
        qdd = sol.states["qddot"]
        jerk = sol.controls['qdddot']

        t = np.linspace(0, final_time-t0, q.shape[1])
        t = t+t0

        data_obj = pd.DataFrame({'Time': t,
                                 'q1': np.rad2deg(sol.states["q"][0]),
                                 # 'q2': np.rad2deg(sol.states["q"][1]),
                                 'dq1': sol.states["qdot"][0],
                                 # 'dq2': sol.states["qdot"][1]
                                 'ddq1': sol.states["qddot"][0],
                                 'dddq1': sol.controls['qdddot'][0]
                                 })
        data_obj.loc[-2] = [0,30,0,0,0]  # adding a row for 0
        data_obj.loc[-1] = [t0/2, 30, 0, 0, 0]
        data_obj.index = data_obj.index + 2  # shifting index
        data_obj = data_obj.sort_index()
        data_obj = data_obj.assign(cost=ifunc)
        all_sol.append(data_obj)

        # biorbd_viz = bioviz.Viz(model_path)
        # biorbd_viz.load_movement(q)
        # biorbd_viz.exec()

        if inside_plot:
            plot_kinematics(t, q, qd, qdd, jerk, f'Q1DOF_v1_overview_{ifunc}.svg')

    if inside_plot:
        data = pd.concat(all_sol)
        var_list = ['Angular position [°]', 'Angular Velocity [rad/s]']
        plot_q_qdot_prx_dist(data, var_list, 'Q1DOF_bio_v1_q_dq.svg' )

    model = biorbd.Model(model_path)
    marker_idx = 0

    t = data_obj.Time.values
    marker_position = pd.DataFrame(data=np.zeros((t.shape[0]*len(all_functions), 7)),
                                   columns=['Time', 'Elbow_x', 'Elbow_y', 'Elbow_d',
                                            'Elbow_vel', 'Elbow_acc','cost'])
    marker_position.Time = np.tile(t, len(all_functions))

    for i, irow in marker_position.iterrows():
        i_obj = i//t.shape[0]
        i_frame = i%t.shape[0]
        pos = model.marker(np.deg2rad(all_sol[i_obj][['q1']].values.T)[:, i_frame], 0).to_array()[:-1]
        s = np.deg2rad(all_sol[i_obj][['q1']].values.T)*0.3
        s = [-s[0][i_frame]+s[0][0]]
        marker_dot = np.linalg.norm(
            model.markersVelocity(
                np.deg2rad(all_sol[i_obj][['q1']].values.T)[:, i_frame],
                all_sol[i_obj][['dq1']].values.T[:, i_frame],
            )[0].to_array())
        marker_ddot = np.linalg.norm(
            model.markerAcceleration(
                np.deg2rad(all_sol[i_obj][['q1']].values.T)[:, i_frame],
                all_sol[i_obj][['dq1']].values.T[:, i_frame],
                all_sol[i_obj][['ddq1']].values.T[:, i_frame],
            )[0].to_array())

        marker_position.loc[i] = np.hstack(([irow.Time],pos, s, marker_dot, marker_ddot, [i_obj]))
    marker_position.cost = np.repeat(all_functions, t.shape[0])

    if external_use:
        return  marker_position, all_sol, [i['cost_value_weighted'] for i in sol.detailed_cost]

    # Evaluate SNR for each
    if evaluate_SNR:
        for icost in ['Minimize jerk',
                      'Minimize jerk and time',
                      'Minimize jerk, energy and time']:
            data_for_snr = marker_position[['Time', 'Elbow_vel', 'Elbow_acc', 'cost']]
            data_for_snr = data_for_snr[data_for_snr.cost == icost]
            param_log_snr = inflexion_points_logparam_robust_t0_fixed(t, data_for_snr.Elbow_vel.values,
                                                                  np.gradient(data_for_snr.Elbow_vel.values, t),
                                                                  [0.1, 0.6, 0.1],
                                                                  t0=t0-t0)
            log_for_cost = param_log_snr[2][0]*lognpdf(t,param_log_snr[0][0], param_log_snr[1][0],t0 = param_log_snr[3][0])
            snr_value = snr(data_for_snr.Elbow_vel.values, log_for_cost)
            print(f'For {icost}, SNR = {snr_value}')

    # to fit log to third obj sol
    if inside_plot:
        to_fit_log = marker_position[['Time', 'Elbow_vel', 'Elbow_acc', 'cost']]
        to_fit_log = to_fit_log[to_fit_log.cost=='Minimize jerk, energy and time']
        # param_log = inflexion_points_logparam_robust(t, to_fit_log.Elbow_vel.values,
        #                                   np.gradient(to_fit_log.Elbow_vel.values, t),[-1.5, 0.6, 1e-2])
        param_log = inflexion_points_logparam_robust_t0_fixed(t, to_fit_log.Elbow_vel.values,
                                                              np.gradient(to_fit_log.Elbow_vel.values, t),
                                                              [0.1, 0.6, 0.1],
                                                              t0=t0-t0)
        log_output = f'The lognormal is defined with D:{np.round(param_log[2][0], 2)}, mu:' \
                     f'{np.round(param_log[0][0],2)},sigma' \
                     f':{np.round(param_log[1][0],2)} and t0: {np.round(param_log[3][0],3)}'
        print(log_output)
        effectors_kin = pd.concat([marker_position[['Time', 'Elbow_vel', 'Elbow_d', 'cost']],
                              pd.DataFrame({'Time':t,
                                            'Elbow_vel':param_log[2]*lognpdf(t,
                                               param_log[0], param_log[1],t0 = param_log[3]),
                                            'cost': ['Lognormal profile']*t.shape[0]})])
        final_time = marker_position.Time.max()
        effectors_kin = effectors_kin.melt(['Time', 'cost'])
        effectors_kin['Which'] = [i.split('_')[-1] for i in effectors_kin.variable]
        g2 = sns.relplot(data=effectors_kin, x='Time', y='value', row='Which', hue='cost',
                         kind='line', row_order=['d', 'vel'], facet_kws={'sharey': False, 'sharex': True},
                         height=2, aspect=2)
        g2.map(plt.axvline, x=t0, color=".9", dashes=(2, 1), zorder=0)
        var_list = ['Displacement norm [m]', 'Velocity norm [m.s⁻¹]']
        t_names = ['max velocity', 'max acc', 'min acc', 'bound', 'bound']

        for i, ax in enumerate(g2.axes.flatten()):
            if i == 0:
                ax.set_ylim(0, 0.6)
                ax.set_yticks([0, 0.3, 0.6])
            else:
                ax.set_ylim(-0.1, 1.9)
                ax.set_yticks([0, 0.9, 1.8])
                for ii, it in enumerate(param_log[-1]):
                    plt.axvline(it, color=".7", dashes=(2, 1), zorder=0)
                    plt.text(it, 1.1, t_names[ii], transform=ax.get_xaxis_transform(), rotation=45)
            ax.set_ylabel(var_list[i])
            ax.set_title(ax.get_title().split()[-1][:-4])
            ax.set_xlabel('Time [s]')
            ax.set_xlim(0, final_time)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
            ax.ticklabel_format(style='sci', axis='y')
            ax.spines[['right', 'top']].set_visible(True)
        plt.savefig('Q1DOF_bio_v1_effectors.svg')
        plt.close()


if __name__ == "__main__":
    main(n_shooting = 200,
         final_time = 0.8,
         cost_function_index = [0, 1, 2],
         alpha_beta=5000,
         inside_plot = True,
         external_use=False,
         evaluate_SNR=True)
