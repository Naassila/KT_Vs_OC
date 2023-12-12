import platform

import numpy as np
import biorbd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import bioviz
from casadi import MX, SX, vertcat, sum1, sum2
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    ConfigureProblem,
    PenaltyController,
    DynamicsFunctions,
    Objective,
    MultinodeObjectiveList,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
    InitialGuessList,
    InterpolationType,
    Node,
)


def min_marker_velocity_dt2(controller: PenaltyController, marker: str) -> MX:
    """
    This function mimics the ObjectiveFcn.MINIMIZE_MARKERSVELOCITY x t^2
    """
    dt = controller.tf / controller.ns
    model = controller.model
    marker_idx = model.marker_index(marker)

    t_node = controller.get_nlp.node_time(controller.node_index) + dt / 2  # to avoid infinity at first node
    state = controller.states
    marker_dot = model.marker_velocities(state["q"].cx, state["qdot"].cx)[marker_idx]
    # marker_dot += model.marker_velocities(state["q"].cx_end, state["qdot"].cx_end)[marker_idx]
    # marker_dot /= 2

    out = sum1(marker_dot**2) * t_node**2 * dt  # todo: valider l'expression

    return out


def min_qdot_dt2(controller: PenaltyController) -> MX:
    """
    This function mimics the ObjectiveFcn.MINIMIZE_STATES x t^2
    """
    dt = controller.tf / controller.ns
    t_node = controller.get_nlp.node_time(controller.node_index) + dt / 2  # to avoid infinity at first node
    qdot = controller.states["qdot"].cx

    out = sum1(qdot**2) * t_node**2 * dt

    return out


def min_jerk_dt2(controller: PenaltyController) -> MX:
    """
    This function mimics the ObjectiveFcn.MINIMIZE_CONTROL * t^2
    """
    dt = controller.tf / controller.ns
    t_node = controller.get_nlp.node_time(controller.node_index) + dt / 2  # to avoid infinity at first node
    jerk = controller.controls["qdddot"].cx

    out = sum1(jerk**2) * t_node**2 * dt

    return out


def multinode_min_marker_jerk(controllers: list[PenaltyController], marker: str) -> MX:
    """
    This function mimics the ObjectiveFcn.MINIMIZE_MARKERSVELOCITY with a multinode objective.
    """
    dt = controllers[0].tf / controllers[0].ns
    model = controllers[0].model
    marker_idx = model.marker_index(marker)

    state_i = controllers[0].states
    state_j = controllers[1].states

    marker_acc_i = model.marker_accelerations(
        state_i["q"].cx,
        state_i["qdot"].cx,
        state_i["qddot"].cx,
    )[marker_idx]
    marker_acc_j = model.marker_accelerations(
        state_j["q"].cx,
        state_j["qdot"].cx,
        state_j["qddot"].cx,
    )[marker_idx]
    marker_jerk = (marker_acc_j - marker_acc_i) / dt
    t_node = controllers[0].get_nlp.node_time(controllers[0].node_index) + dt / 2  # to avoid infinity at first node

    return sum1(marker_jerk**2) * t_node**2 * dt  # todo: valider l'expression


def custom_dynamics(
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    stochastic_variables: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

    Parameters
    ----------
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    qddot = DynamicsFunctions.get(nlp.states["qddot"], states)

    jerk = DynamicsFunctions.get(nlp.controls["qdddot"], controls)

    # call bioptim accessor
    dq = qdot  # DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = qddot  # DynamicsFunctions.compute_qddot(nlp, qdot, qddot)
    dddq = jerk

    # the user has to choose if want to return the explicit dynamics dx/dt = f(x,u,p)
    # as the first argument of DynamicsEvaluation or
    # the implicit dynamics f(x,u,p,xdot)=0 as the second argument
    # which may be useful for IRK or COLLOCATION integrators

    return DynamicsEvaluation(dxdt=vertcat(dq, ddq, dddq), defects=None)  # todo: prendre les defects si DC


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    my_additional_factor: int
        An example of an extra parameter sent by the user
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qddot(ocp, nlp, as_states=True, as_controls=False)

    ConfigureProblem.configure_qdddot(ocp, nlp, as_states=False, as_controls=True)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)


def prepare_ocp(
    biorbd_model_path: str,
    weights: float,
    n_shooting: int,
    final_time: float,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    assume_phase_dynamics: bool = False,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    ode_solver: OdeSolverBase
        The type of ode solver used
    use_sx: bool
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamics, expand=True)

    # Add objective functions
    objective_functions = ObjectiveList()
    multinode_objectives = MultinodeObjectiveList()


    obj_func_keys = ['Minimize end effector velocity * t (Mayer)', 'Minimize end effector jerk * t (mn)',
                     'Minimize end effector velocity (Lgr)', 'Minimize end effector acceleration (Lgr)',
                     'Minimize qdot * t (Mayer)', 'Minimize qjerk * t (Mayer)',
                     'Minimize qdot (Lgr)', 'Minimize qjerk (Lgr)']
    weights_dict = dict(zip(obj_func_keys, weights))
    print(weights_dict)

    for i in range(n_shooting):
        objective_functions.add(
            min_marker_velocity_dt2,
            marker="wrist",
            custom_type=ObjectiveFcn.Mayer,
            node=i,
            quadratic=False,
            weight=weights[0],
            expand=True,
        )


    for i in range(n_shooting):
        multinode_objectives.add(
            multinode_min_marker_jerk,
            marker="wrist",
            nodes_phase=[0, 0],
            nodes=[i, i + 1],
            weight=weights[1],
            quadratic=False,
            expand=True,
        )


    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_MARKERS_VELOCITY,
        marker_index="wrist",
        node=Node.ALL,
        weight=weights[2],
        quadratic=True,
        expand=True,
        derivative=False,
    ),


    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_MARKERS_ACCELERATION,
        marker_index="wrist",
        weight=weights[3],
        quadratic=True,
        expand=True,
        derivative=True,
    ),


    for i in range(n_shooting):
        objective_functions.add(
            min_qdot_dt2,
            custom_type=ObjectiveFcn.Mayer,
            node=i,
            quadratic=False,
            weight=weights[4],
        )


    for i in range(n_shooting):
        objective_functions.add(
            min_jerk_dt2,
            custom_type=ObjectiveFcn.Mayer,
            node=i,
            quadratic=False,
            weight=weights[5],
        )


    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="qdot",
        node=Node.ALL,
        weight=0,
        quadratic=True,
        expand=True,
        derivative=False,
    ),


    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        node = Node.ALL_SHOOTING,
        key="qdddot",
        weight=0,
        quadratic=True,
        expand=True,
        derivative=False,
    ),

    # Constraints
    # constraints = ConstraintList()

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

    x_bounds["qdot"] = [-10, -10], [10, 10]
    x_bounds["qddot"] = [-50, -50], [50, 50]

    x_bounds["qdot"][:, [0, -1]] = 0.0
    x_bounds["qddot"][:, [0, -1]] = 0.0

    # Define control path constraint
    jerk_min, jerk_max = -200, 200
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

    # [0] min_marker_velocity_dt2;              [1] multinode_min_marker_jerk
    # [2] Lagrange.MINIMIZE_MARKERS_VELOCITY    [3] Lagrange.MINIMIZE_MARKERS_ACCELERATION
    # [4] min_qdot_dt2                          [5] min_jerk_dt2
    # [6] Lagrange.MINIMIZE_STATE               [7] Lagrange.MINIMIZE_CONTROL
    weights = np.zeros((8, 1))
    weights[0] = 0
    weights[1] = 0
    weights[5] = 5000

    # Problem parameters
    n_shooting = 60
    final_time = 0.7

    ocp = prepare_ocp(biorbd_model_path=model_path, weights=weights,
                      final_time=final_time,
                      n_shooting=n_shooting)

    # --- Solve the program --- #

    solver_ipopt = Solver.IPOPT()
    solver_ipopt.set_linear_solver("ma57")

    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))
    print(sol.status)

    # --- Show results --- #
    #    sol.animate()
    # sol.graphs(show_bounds=True)
    # sol.print_cost()

    q = sol.states["q"]
    qd = sol.states["qdot"]
    t = np.linspace(0, final_time, q.shape[1])

    data = pd.DataFrame({'Time': t,
                        'q1': np.rad2deg(sol.states["q"][0]),
                         'q2': np.rad2deg(sol.states["q"][1]),
                         'dq1': sol.states["qdot"][0],
                         'dq2': sol.states["qdot"][1]})

    biorbd_viz = bioviz.Viz(model_path)
    biorbd_viz.load_movement(q)
    biorbd_viz.exec()

    df = data.melt('Time')
    df['Type'] = [i[0] for i in df.variable]
    df['Joint'] = [i[-2:] for i in df.variable]

    g = sns.relplot(data=df, x='Time', y='value', row='Type', col='Joint', kind='line',
                    facet_kws = {'sharey': False, 'sharex': True},
                    height = 2, aspect = 1.5
                    )

    var_list = ['Angular position [°]', 'Angular Velocity [rad/s]']
    g.axes[0][0].set_ylabel(var_list[0])
    g.axes[0][0].set_title('Proximal joint')
    g.axes[0][1].set_title('Distal joint')

    g.axes[1][0].set_ylabel(var_list[1])
    g.axes[1][0].set_title('')

    g.axes[1][1].set_title('')
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_xlabel('Time [s]')
        ax.set_xlim(0, final_time)
        # ax.axvline(0.1, color='k', linestyle=':', alpha=0.5)
        # ax.axvline(0.2, color='k', linestyle=':', alpha=0.5)
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.ticklabel_format(style='sci', axis='y')
        ax.spines[['right', 'top']].set_visible(True)
    plt.savefig('bio_test.png')
    plt.close()


    model = biorbd.Model(model_path)
    marker_idx = 1  # model.marker_index("wrist")
    wrist_dot = []
    wrist_dot_norm = []

    marker_position = pd.DataFrame(data=np.zeros((q.shape[1], 9)),
                                   columns = ['Time', 'Elbow_x', 'Elbow_y', 'Wrist_x', 'Wrist_y',
                                              'Wrist_x_vel', 'Wrist_velocity_y_vel',
                                              'Wrist_vel', 'Elbow_vel'])
    marker_position.Time = t

    for i, irow in marker_position.iterrows():
        pos = np.hstack([model.marker(q[:, i], j).to_array()[:-1] for j in range(model.nbMarkers())])
        wrist_vel = model.markersVelocity(q[:, i], qd[:, i])[marker_idx].to_array()[:-1]
        marker_dot = np.hstack([np.linalg.norm(model.markersVelocity(q[:, i], qd[:, i])[j].to_array()) for j in range(model.nbMarkers())])
        irow[1:] = np.hstack((pos, wrist_vel, marker_dot))

    effectors_velocity = marker_position[['Time', 'Wrist_vel', 'Elbow_vel']].melt('Time')
    g2 = sns.relplot(data=effectors_velocity, x='Time', y='value', col='variable',
                     kind='line', col_order=['Elbow_vel', 'Wrist_vel'], facet_kws={'sharey': False, 'sharex': True},
                     height=2, aspect=1.5)
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
    plt.savefig('bio_test_effectors_vel.png')
    plt.close()

    data_smooth = pd.concat([
        marker_position[['Time','Elbow_x', 'Elbow_y',]].rename(columns={'Elbow_x':'x', 'Elbow_y':'y'}).assign(Joint='Elbow'),
        marker_position[['Time','Wrist_x', 'Wrist_y',]].rename(columns={'Wrist_x':'x', 'Wrist_y':'y'}).assign(Joint='Wrist')
    ])

    g3 = sns.relplot(data=data_smooth, x='x', y='y', style='Joint', height=3, aspect=1, markers=['4', '4'],
                     color='grey', alpha=0.5,
                     # col_order=['Minimize jerk', 'Minimize jerk and time', 'Minimize jerk, energy and time', ],
                     legend=False)
    data=marker_position[['Time','Elbow_x', 'Elbow_y','Wrist_x', 'Wrist_y',]]
    data.Time = data.Time.round(3)
    data = data.loc[
        (data.Time == 0) |
        (data.Time == 0.1) |
        (data.Time == 0.2) |
        (data.Time == final_time/2) |
        (data.Time == final_time)]

    ax = g3.axes[0][0]

    for iirow, irow in data.iterrows():
        ax.plot([0, irow.Elbow_x, irow.Wrist_x], [0, irow.Elbow_y, irow.Wrist_y], color='k',
                alpha=0.4 + iirow * 0.2 / 104)

    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(-0.1, 0.8)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.spines[['right', 'top']].set_visible(True)

    plt.savefig('bio_test_arm.png')
    plt.close()

    fig2, ax2 = plt.subplots()
    ax2.plot(wrist_dot)
    ax2.plot(wrist_dot_norm)
    plt.show()


if __name__ == "__main__":
    main()
