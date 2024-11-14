from dataclasses import dataclass, field
from typing import Union

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)

from pdm4ar.exercises.ex11.discretization import *
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp

        # Solver Parameters
        self.params = SolverParameters()

        self.s_prime = {name: cvx.Variable((self.params.K, 1), nonneg=True) for name in planets.keys()}

        # Spaceship Dynamics
        self.spaceship = SpaceshipDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # Initialize with zeros
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SpaceshipState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        # TODO: Implement this method
        pass

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K
        X = np.zeros((self.spaceship.n_x, K))
        U = np.zeros((self.spaceship.n_u, K))
        p = np.array([15.0])  # Initial time guess
        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K)),
            "p": cvx.Variable(self.spaceship.n_p),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x),
            "goal_state": cvx.Parameter(self.spaceship.n_x),
            "X_last": cvx.Parameter((self.spaceship.n_x * self.spaceship.n_x, self.params.K)),
            "U_last": cvx.Parameter((self.spaceship.n_u * self.spaceship.n_x, self.params.K)),
            "A_bar": cvx.Parameter((self.spaceship.n_x, self.spaceship.n_x, self.params.K - 1)),
            "B_bar_plus": cvx.Parameter((self.spaceship.n_x, self.spaceship.n_u, self.params.K - 1)),
            "B_bar_minus": cvx.Parameter((self.spaceship.n_x, self.spaceship.n_u, self.params.K - 1)),
            "C_bar": cvx.Parameter((self.spaceship.n_x, self.spaceship.n_u, self.params.K - 1)),
            "F_bar": cvx.Parameter((self.spaceship.n_x, 1, self.params.K - 1)),
            "p_last": cvx.Parameter(self.spaceship.n_p),
            "r_bar": cvx.Parameter((self.spaceship.n_x, 1, self.params.K - 1)),
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """Define constraints for SCvx."""
        X = self.variables["X"]  # state variables [n_x, K]
        U = self.variables["U"]  # control inputs [n_u, K]
        p = self.variables["p"]  # time parameter
        K = self.params.K

        # Initial and Final constraints
        constraints = [
            # Initial state
            X[:, 0] == self.problem_parameters["init_state"],
            # Final state
            X[0:6, -1] == self.problem_parameters["goal_state"],
            # Input constraints
            U[:, 0] == 0,  # zero initial input
            U[:, -1] == 0,  # zero final input
        ]

        # State constrains
        constraints += [
            X[7, :] >= self.sp.m_v,  # mass above vehicle mass
            X[6, :] >= self.sp.delta_limits[0],  # min steering angle
            X[6, :] <= self.sp.delta_limits[1],  # max steering angle
        ]

        # Input constraints
        constraints += [
            U[0, :] >= self.sp.thrust_limits[0],  # min thrust
            U[0, :] <= self.sp.thrust_limits[1],  # max thrust
            U[1, :] >= self.sp.ddelta_limits[0],  # min steering
            U[1, :] <= self.sp.ddelta_limits[1],  # max steering
        ]

        # Time constraints
        constraints += [
            p >= 0.0,  # time positive
        ]

        for planet_name, planet in self.planets.items():
            p_c = planet.center
            min_dist = planet.radius + self.sg.l

            for k in range(K):
                constraints += [cvx.norm(X[0:2, k] - p_c) >= min_dist]

        # # Obstacles - Satellites
        # for sat in self.satellites.values():
        #     radius = sat.radius + self.sg.l
        #     for k in range(K):
        #         t = k * p / (K - 1)

        #         # Linear approximation of satellite position
        #         angle = sat.tau + sat.omega * t
        #         ca = np.cos(sat.tau)  # Fixed angle component
        #         sa = np.sin(sat.tau)

        #         x_sat = sat.orbit_r * (ca - t * sat.omega * sa)
        #         y_sat = sat.orbit_r * (sa + t * sat.omega * ca)
        #         sat_pos = np.array([x_sat, y_sat])

        #         # Squared distance constraint
        #         diff = X[:2, k] - sat_pos
        #         constraints.append(cvx.sum_squares(diff) >= radius * radius)

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]

        # Minimize time
        time_obj = self.params.weight_p @ p

        # Minimize fuel consumption (mass loss)
        init_mass = X[7, 0]
        final_mass = X[7, -1]
        fuel_obj = init_mass - final_mass

        # Minimize control effort
        control_obj = cvx.sum_squares(U[0, :])  # thrust
        steering_obj = cvx.sum_squares(U[1, :])  # steering

        # Weights
        w_time = 1.0
        w_fuel = 0.1
        w_control = 0.01
        w_steering = 0.01

        objective = w_time * time_obj + w_fuel * fuel_obj + w_control * control_obj + w_steering * steering_obj

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        self.problem_parameters["A_bar"].value = A_bar
        self.problem_parameters["B_plus_bar"].value = B_plus_bar  # For FOH
        self.problem_parameters["B_minus_bar"].value = B_minus_bar  # For FOH
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar
        self.problem_parameters["X_last"].value = self.X_bar
        self.problem_parameters["U_last"].value = self.U_bar
        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        self.problem_parameters["goal_state"].value = self.goal.as_ndarray()

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """

        pass

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        pass

    @staticmethod
    def _extract_seq_from_array() -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F = np.array([0, 1, 2, 3, 4])
        ddelta = np.array([0, 0, 0, 0, 0])
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 8)
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)
        return mycmds, mystates
