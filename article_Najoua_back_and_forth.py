import platform
import bioviz
from bioptim import OdeSolverBase, PhaseTransitionFcn, PhaseTransitionList, BiMapping
from Biotiom_post_tools import *
from Bioptim_functions_ocp import *

def prepare_ocp(
    biorbd_model_path: str,
    n_shooting:int,
    final_time:float,
    cost_dt2: bool = True,
    cost_marker: bool = True,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    assume_phase_dynamics: bool = False,
) -> OptimalControlProgram:

    # --- Options --- #
    # BioModel path
    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamics, expand=True)
    dynamics.add(custom_configure, dynamic_function=custom_dynamics, expand=True)

    # Add objective functions
    objective_functions = ObjectiveList()
    multinode_objectives_1 = MultinodeObjectiveList()
    multinode_objectives_2 = MultinodeObjectiveList()
    multinode_objectives = MultinodeObjectiveList()

    if cost_dt2 and cost_marker:
        for i in range(n_shooting[0]):
            # end_effector_velocity * t
            for iphase in range(len(n_shooting)):
                objective_functions.add(
                    min_marker_velocity_dt2,
                    marker="wrist",
                    custom_type=ObjectiveFcn.Mayer,
                    node=i,
                    quadratic=False,
                    weight=5000,
                    expand=True,
                    phase = iphase,
                )

        for i in range(n_shooting[0]):
            # end_effector_jerk * t
            multinode_objectives.add(
                multinode_min_marker_jerk_dt2,
                marker="wrist",
                nodes_phase=[0, 0],
                nodes=[i, i + 1],
                weight=1,
                quadratic=False,
                expand=True,
                # phase=0,
            )
        for i in range(n_shooting[1]):
            # end_effector_jerk * t
            multinode_objectives.add(
                multinode_min_marker_jerk_dt2,
                marker="wrist",
                nodes_phase=[1, 1],
                nodes=[i, i + 1],
                weight=1,
                quadratic=False,
                expand=True,
            )

    elif not cost_dt2 and cost_marker:
        # end_effector_velocity
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_MARKERS_VELOCITY,
            marker_index="wrist",
            node=Node.ALL,
            weight=1000,
            phase=0,
            quadratic=True,
            expand=True,
            derivative=False,
        ),
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_MARKERS_VELOCITY,
            marker_index="wrist",
            node=Node.ALL,
            weight=1000,
            phase=1,
            quadratic=True,
            expand=True,
            derivative=False,
        ),

        # end_effector_jerk
        for i in range(n_shooting[0]):
            # end_effector_jerk
            multinode_objectives.add(
                multinode_min_marker_jerk,
                marker="wrist",
                nodes_phase=[0, 0],
                nodes=[i, i + 1],
                weight=1,
                phase=0,
                quadratic=False,
                expand=True,
            )
        for i in range(n_shooting[1]):
            # end_effector_jerk
            multinode_objectives.add(
                multinode_min_marker_jerk,
                marker="wrist",
                nodes_phase=[0, 0],
                nodes=[i, i + 1],
                weight=1,
                phase=1,
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

    elif cost_dt2 and not cost_marker:
        for i in range(n_shooting[0]):
            # angular_velocity * t
            objective_functions.add(
                min_qdot_dt2(),
                custom_type=ObjectiveFcn.Mayer,
                node=i,
                quadratic=False,
                weight=1000,
                phase=0,
            )
        for i in range(n_shooting[1]):
            # angular_velocity * t
            objective_functions.add(
                min_qdot_dt2(),
                custom_type=ObjectiveFcn.Mayer,
                node=i,
                quadratic=False,
                weight=1000,
                phase=1,
            )

        for i in range(n_shooting[0]):
            # angular_jerk * t
            objective_functions.add(
                min_jerk_dt2(),
                custom_type=ObjectiveFcn.Mayer,
                node=i,
                quadratic=False,
                weight=1,
                phase=0,
            )
        for i in range(n_shooting[1]):
            # angular_jerk * t
            objective_functions.add(
                min_jerk_dt2(),
                custom_type=ObjectiveFcn.Mayer,
                node=i,
                quadratic=False,
                weight=1,
                phase=1,
            )

    elif not cost_dt2 and not cost_marker:
        # angular_velocity
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="qdot",
            node=Node.ALL,
            weight=1000,
            phase = 0,
            quadratic=True,
            expand=True,
            derivative=False,
        ),
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="qdot",
            node=Node.ALL,
            weight=1000,
            phase=1,
            quadratic=True,
            expand=True,
            derivative=False,
        ),

        # angular_jerk
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qdddot", weight=1, phase=0, quadratic=True, expand=True, derivative=True
        ),
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qdddot", weight=1, phase=1, quadratic=True, expand=True, derivative=True
        ),

    # Path constraint
    d2r = np.pi / 180
    x_bounds = BoundsList()
    x_bounds.add("q", bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("q", bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qddot", bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qddot", bio_model[1].bounds_from_ranges("q"), phase=1)

    # start position
    x_bounds[0]["q"][0, 0] = 30 * d2r
    x_bounds[0]["q"][1, 0] = 60 * d2r

    # mid position
    x_bounds[1]["q"][0, 0] = -70 * d2r
    x_bounds[1]["q"][1, 0] = 10 * d2r
    # x_bounds[0]["q"][0, -1] = -70 * d2r
    # x_bounds[0]["q"][1, -1] = 10 * d2r

    # end position
    x_bounds[1]["q"][0, -1] = 30 * d2r
    x_bounds[1]["q"][1, -1] = 60 * d2r

    x_bounds[0]["qdot"] = [-20, -20], [20, 20]
    x_bounds[1]["qdot"] = [-20, -20], [20, 20]
    x_bounds[0]["qdot"][:, 0] = 0.0
    x_bounds[1]["qdot"][:, -1] = 0.0

    x_bounds[0]["qddot"] = [-1000, -1000], [1000, 1000]
    x_bounds[1]["qddot"] = [-1000, -1000], [1000, 1000]
    x_bounds[0]["qddot"][:, 0] = 0.0
    x_bounds[1]["qddot"][:, -1] = 0.0

    # Define control path constraint
    jerk_min, jerk_max = -1e30, 1e30
    u_bounds = BoundsList()
    u_bounds.add("qdddot", min_bound=[jerk_min] * bio_model[0].nb_tau, max_bound=[jerk_max] * bio_model[0].nb_tau, phase=0)
    u_bounds.add("qdddot", min_bound=[jerk_min] * bio_model[1].nb_tau, max_bound=[jerk_max] * bio_model[1].nb_tau, phase=1)
    # Initial guess (optional since it is 0, we show how to initialize anyway)
    #    x_init = InitialGuessList()
    #    x_init.add("q", min_bound=[10, 140], max_bound=[70, 20], interpolation=InterpolationType.LINEAR)

    # ------------- #

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0, states_mapping=BiMapping([-1], [0]))

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
    n_shooting = [60,60]
    final_time = [0.8, 1.6]

    ocp = prepare_ocp(biorbd_model_path=model_path, cost_dt2=True, cost_marker=True,
                      n_shooting = n_shooting, final_time=final_time)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    sol.graphs()
    sol.animate()
    print(f'Solver ended with {sol.status}')
    plot_cost(sol, n_shooting, 'bio_v1_costs_BF.png')
    q = np.hstack((sol.states[0]["q"], sol.states[1]["q"]))
    qd = np.hstack((sol.states[0]["qdot"], sol.states[1]["qdot"]))
    qdd = np.hstack((sol.states[0]["qddot"], sol.states[1]["qddot"]))
    jerk = np.hstack((sol.controls[0]["qdddot"], sol.controls[1]["qdddot"]))

    t = np.linspace(0, np.sum(final_time), q.shape[1])

    data = pd.DataFrame({'Time': t,
                         'q1': np.rad2deg(q[0]),
                         'q2': np.rad2deg(q[1]),
                         'dq1': qd[0],
                         'dq2': qd[1]})

    biorbd_viz = bioviz.Viz(model_path)
    biorbd_viz.load_movement(q)
    biorbd_viz.exec()

    plot_kinematics(t, q, qd, qdd, jerk, 'v1_overview_BF.png')

    var_list = ['Angular position [°]', 'Angular Velocity [rad/s]']
    plot_q_qdot_prx_dist(data, var_list, 'bio_v1_q_dq_BF.png' )

    model = biorbd.Model(model_path)
    marker_idx = 1  # model.marker_index("wrist")

    marker_position = pd.DataFrame(data=np.zeros((q.shape[1], 9)),
                                   columns=['Time', 'Elbow_x', 'Elbow_y', 'Wrist_x', 'Wrist_y',
                                            'Wrist_x_vel', 'Wrist_velocity_y_vel',
                                            'Wrist_vel', 'Elbow_vel'])
    marker_position.Time = t

    for i, irow in marker_position.iterrows():
        pos = np.hstack([model.marker(q[:, i], j).to_array()[:-1] for j in range(model.nbMarkers())])
        wrist_vel = model.markersVelocity(q[:, i], qd[:, i])[marker_idx].to_array()[:-1]
        marker_dot = np.hstack(
            [np.linalg.norm(model.markersVelocity(q[:, i], qd[:, i])[j].to_array()) for j in range(model.nbMarkers())])
        irow[1:] = np.hstack((pos, wrist_vel, marker_dot))

    plot_effector_velocity(marker_position, 'bio_v1_effectors_vel_BF.png')
    plot_arm_in_time(marker_position, 'bio_v1_arm_BF.png')

if __name__ == "__main__":
    main()
