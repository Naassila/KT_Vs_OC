import platform
import bioviz
from bioptim import OdeSolverBase
from Biotiom_post_tools import *
from Bioptim_functions_ocp import *

def prepare_ocp(
    biorbd_model_path: str,
    n_shooting:int,
    final_time:float,
    cost_dt2: bool = True,
    cost_marker: bool = True,
    min_jerk: bool = True,
    min_energy: bool = True,
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
                weight=1e5,
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
                quadratic=False,
                weight=1e5,
            )
        for i in range(n_shooting):
            # angular_jerk * t
            objective_functions.add(
                min_jerk_dt2,
                custom_type=ObjectiveFcn.Mayer,
                node=i,
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

    x_bounds["qdot"] = [-50, -50], [50, 50]
    x_bounds["qddot"] = [-2000, -2000], [2000, 2000]

    x_bounds["qdot"][:, [0, -1]] = 0.0  # qdot = 0 at start and end
    x_bounds["qddot"][:, [0, -1]] = 0.0  # qDdot = 0 at start and end

    # Define control path constraint
    jerk_min, jerk_max = -1e30, 1e30
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


def main():
    """
    Runs the optimization and animates it
    """

    model_path = "arm.bioMod"

    # Problem parameters
    n_shooting = 120
    final_time = 0.3
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
                          final_time=final_time)

        # --- Solve the program --- #
        sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
        print(f'Solver for {ifunc} ended with {sol.status}')
        if ifunc == 'Minimize jerk, energy and time':
            plot_cost(sol, n_shooting, f'Q_bio_v1_costs_{ifunc}.png')
        q = sol.states["q"]
        qd = sol.states["qdot"]
        qdd = sol.states["qddot"]
        jerk = sol.controls['qdddot']

        t = np.linspace(0, final_time, q.shape[1])

        data_obj = pd.DataFrame({'Time': t,
                                 'q1': np.rad2deg(sol.states["q"][0]),
                                 'q2': np.rad2deg(sol.states["q"][1]),
                                 'dq1': sol.states["qdot"][0],
                                 'dq2': sol.states["qdot"][1]}).assign(cost=ifunc)
        all_sol.append(data_obj)

        # biorbd_viz = bioviz.Viz(model_path)
        # biorbd_viz.load_movement(q)
        # biorbd_viz.exec()

        plot_kinematics(t, q, qd, qdd, jerk, f'Q_v1_overview_{ifunc}.svg')

    data = pd.concat(all_sol)
    var_list = ['Angular position [°]', 'Angular Velocity [rad/s]']
    plot_q_qdot_prx_dist(data, var_list, 'Q_bio_v1_q_dq.svg' )

    model = biorbd.Model(model_path)
    marker_idx = 1  # model.marker_index("wrist")

    marker_position = pd.DataFrame(data=np.zeros((q.shape[1]*3, 10)),
                                   columns=['Time', 'Elbow_x', 'Elbow_y', 'Wrist_x', 'Wrist_y',
                                            'Wrist_x_vel', 'Wrist_velocity_y_vel',
                                            'Elbow_vel', 'Wrist_vel', 'cost'])
    marker_position.Time = np.hstack((t,t,t))

    for i, irow in marker_position.iterrows():
        i_obj = i//q.shape[1]
        i_frame = i%q.shape[1]
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
    marker_position.cost = np.repeat(list(obj_dict.keys()), q.shape[1])

    plot_effector_velocity(marker_position, 'Q_bio_v1_effectors_vel.svg')
    plot_arm_in_time(marker_position,
                     ['Minimize jerk', 'Minimize jerk and time', 'Minimize jerk, energy and time', ],
                     'Q_bio_v1_arm.svg')

if __name__ == "__main__":
    main()
