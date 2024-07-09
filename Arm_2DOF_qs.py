import platform
import bioviz
from bioptim import OdeSolverBase
from Biotiom_post_tools import *
from Bioptim_functions_ocp import *
from Tools import lognpdf, inflexion_points_logparam_robust_t0_fixed, snr

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
                command_delay =t0,
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
    x_bounds["q"][1, 0] = 60 * d2r
    # end position
    x_bounds["q"][0, -1] = -70 * d2r
    x_bounds["q"][1, -1] = 10 * d2r

    x_bounds["qdot"] = [-600*d2r, -600*d2r], [600*d2r, 600*d2r] #[-50, -50], [50, 50]
    x_bounds["qddot"] = [-5000*d2r, -5000*d2r], [5000*d2r, 5000*d2r] #[-2000, -2000], [2000, 2000]

    x_bounds["qdot"][:, [0, -1]] = 0.0  # qdot = 0 at start and end
    x_bounds["qddot"][:, [0, -1]] = 0.0  # qDdot = 0 at start and end

    # Define control path constraint
    jerk_min, jerk_max = -4000, 4000 #-1e30, 1e30
    u_bounds = BoundsList()
    u_bounds.add("qdddot", min_bound=[jerk_min] * bio_model.nb_tau, max_bound=[jerk_max] * bio_model.nb_tau)

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


def main(n_shooting=120, final_time:float=0.3, alpha_beta:float=5000,  t0:float=0):
    """
    Runs the optimization and animates it
    """

    model_path = "arm.bioMod"

    # Problem parameters
    obj_dict = {'Minimize jerk':{'cost_dt2':False, 'cost_marker':False,
                                 'min_energy': False, 'min_jerk':True},
                'Minimize jerk and time':{'cost_dt2':True, 'cost_marker':False,
                                          'min_energy': False, 'min_jerk':True},
                'Minimize jerk, energy and time':{'cost_dt2':True, 'cost_marker':False,
                                                  'min_energy': True, 'min_jerk':True}, }
    all_sol = []
    for ifunc in ['Minimize jerk', 'Minimize jerk and time',
                   'Minimize jerk, energy and time']:

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
        print(f'Solver for {ifunc} ended with {sol.status}')
        if ifunc == 'Minimize jerk, energy and time':
            plot_cost(sol, n_shooting, f'Q_bio_v1_costs_{ifunc}.png')
        q = sol.states["q"]
        qd = sol.states["qdot"]
        qdd = sol.states["qddot"]
        jerk = sol.controls['qdddot']

        t = np.linspace(0, final_time-t0, q.shape[1])
        t = t+t0

        data_obj = pd.DataFrame({'Time': t,
                                 'q1': np.rad2deg(sol.states["q"][0]),
                                 'q2': np.rad2deg(sol.states["q"][1]),
                                 'dq1': sol.states["qdot"][0],
                                 'dq2': sol.states["qdot"][1]})
        data_obj.loc[-2] = [0,30,60,0,0]  # adding a row for 0
        data_obj.loc[-1] = [t0/2, 30, 60, 0, 0]
        data_obj.index = data_obj.index + 2  # shifting index
        data_obj = data_obj.sort_index()
        data_obj = data_obj.assign(cost=ifunc)
        all_sol.append(data_obj)

        # biorbd_viz = bioviz.Viz(model_path)
        # biorbd_viz.load_movement(q)
        # biorbd_viz.exec()

        # plot_kinematics(t, q, qd, qdd, jerk, f'Q2DOFqs_v1_overview_{ifunc}.svg')

    data = pd.concat(all_sol)
    var_list = ['Angular position [°]', 'Angular Velocity [rad/s]']
    # plot_q_qdot_prx_dist(data, var_list, 'Q2DOFqs_bio_v1_q_dq.svg' )
    # plot_interarticular(data, ['q1 [°]', 'q2 [°]'], 'Q2DOFqs_v1_q1_q2.svg')

    model = biorbd.Model(model_path)
    marker_idx = 1  # model.marker_index("wrist")

    marker_position = pd.DataFrame(data=np.zeros((all_sol[0].shape[0]*3, 10)),
                                   columns=['Time', 'Elbow_x', 'Elbow_y', 'Wrist_x', 'Wrist_y',
                                            'Wrist_x_vel', 'Wrist_velocity_y_vel',
                                            'Elbow_vel', 'Wrist_vel', 'cost'])
    marker_position.Time = np.hstack((all_sol[0].Time.values,all_sol[0].Time.values,all_sol[0].Time.values))

    for i, irow in marker_position.iterrows():
        i_obj = i//all_sol[0].shape[0]
        i_frame = i%all_sol[0].shape[0]
        pos = np.hstack([model.marker(
            np.deg2rad(all_sol[i_obj][['q1', 'q2']].values.T)[:, i_frame], j).to_array()[:-1] for j in range(model.nbMarkers())])
        wrist_vel = model.markersVelocity(
            np.deg2rad(all_sol[i_obj][['q1', 'q2']].values.T[:, i_frame]),
            all_sol[i_obj][['dq1', 'dq2']].values.T[:, i_frame])[marker_idx].to_array()[:-1]
        marker_dot = np.hstack(
            [np.linalg.norm(model.markersVelocity(
                np.deg2rad(all_sol[i_obj][['q1', 'q2']].values.T)[:, i_frame],
                all_sol[i_obj][['dq1', 'dq2']].values.T[:, i_frame])[j].to_array()) for j in range(model.nbMarkers())])
        marker_position.loc[i] = np.hstack(([irow.Time],pos, wrist_vel, marker_dot, [i_obj]))
    marker_position.cost = np.repeat(list(obj_dict.keys()), all_sol[0].shape[0])

    # Evaluate SNR:
    for icost in ['Minimize jerk',
                  'Minimize jerk and time',
                  'Minimize jerk, energy and time']:
        data_for_snr = marker_position[['Time', 'Wrist_vel', 'cost']]
        data_for_snr = data_for_snr[data_for_snr.cost == icost]
        param_log_snr = inflexion_points_logparam_robust_t0_fixed(data_for_snr.Time.values,
                                                                  data_for_snr.Wrist_vel.values,
                                                                  np.gradient(data_for_snr.Wrist_vel.values,
                                                                              data_for_snr.Time.values),
                                                                  [0.1, 0.6, 0.1],
                                                                  t0=t0-t0)
        log_for_cost = param_log_snr[2][0] * lognpdf(data_for_snr.Time.values, param_log_snr[0][0], param_log_snr[1][0],
                                                     t0=param_log_snr[3][0])
        snr_value = snr(data_for_snr.Wrist_vel.values, log_for_cost)
        print(f'For {icost}, SNR = {snr_value}')
        log_output = f'The lognormal is defined with D:{np.round(param_log_snr[2][0], 2)}, mu:' \
                     f'{np.round(param_log_snr[0][0], 2)},sigma' \
                     f':{np.round(param_log_snr[1][0], 2)} and t0: {np.round(param_log_snr[3][0], 3)}'
        print(log_output)

    # plot_effector_velocity(marker_position, 'Q2DOFqs_bio_v1_effectors_vel.svg')
    plot_arm_in_time(marker_position,
                     ['Minimize jerk', 'Minimize jerk and time', 'Minimize jerk, energy and time', ],
                     'Q2DOFqs_bio_v1_arm.svg')

if __name__ == "__main__":
    main(n_shooting=200,
         final_time=0.8,
         alpha_beta=1e5,
         t0=0.01)
