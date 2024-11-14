import sympy as spy

from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters


class SpaceshipDyn:
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SpaceshipGeometry, sp: SpaceshipParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi delta m", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust ddelta", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """Define dynamics for optimization.
        x = [x, y, psi, vx, vy, dpsi, delta, m]
        u = [thrust, ddelta]
        p = [t_f]
        """
        f = spy.zeros(self.n_x, 1)

        # Position
        f[0, 0] = (self.x[3, 0] * spy.cos(self.x[2, 0]) - self.x[4, 0] * spy.sin(self.x[2, 0])) * self.p
        f[1, 0] = (self.x[3, 0] * spy.sin(self.x[2, 0]) + self.x[4, 0] * spy.cos(self.x[2, 0])) * self.p

        # Heading
        f[2, 0] = self.x[5, 0] * self.p

        # Velocity
        f[3, 0] = (1 / self.x[7, 0] * spy.cos(self.x[6, 0]) * self.u[0, 0] + self.x[4, 0] * self.x[5, 0]) * self.p
        f[4, 0] = (1 / self.x[7, 0] * spy.sin(self.x[6, 0]) * self.u[0, 0] - self.x[3, 0] * self.x[5, 0]) * self.p

        # Angular velocity
        f[5, 0] = -self.sg.l_r / self.sg.Iz * spy.sin(self.x[6, 0]) * self.u[0, 0] * self.p
        f[6, 0] = self.u[1, 0] * self.p

        # Fuel
        f[7, 0] = -self.sp.C_T * self.u[0, 0] * self.p

        # Convert to functions with safe evaluation
        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), f.jacobian(self.x), "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), f.jacobian(self.u), "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), f.jacobian(self.p), "numpy")

        return f_func, A_func, B_func, F_func
