import biorbd
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


def multinode_min_marker_jerk_dt2(controllers: list[PenaltyController], marker: str) -> MX:
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

    return sum1(marker_jerk**2)  # todo: valider l'expression
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