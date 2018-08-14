import numpy as np

from copulas.bivariate.base import Bivariate, CopulaTypes


class Clayton(Bivariate):
    """Class for clayton copula model."""

    copula_type = CopulaTypes.CLAYTON

    def get_generator(self):
        """Return the generator function."""

        def generator(theta, t):
            return 1.0 / theta * (np.power(t, -theta) - 1)

        return generator

    def probability_density(self, U, V):
        """Compute density function for given copula family."""
        if self.theta < 0:
            raise ValueError("Theta cannot be less or equal than 0 for Clayton")

        elif self.theta == 0:
            return np.multiply(U, V)

        else:
            a = (self.theta + 1) * np.power(np.multiply(U, V), -(self.theta + 1))
            b = np.power(U, -self.theta) + np.power(V, -self.theta) - 1
            c = -(2 * self.theta + 1) / self.theta
            return a * np.power(b, c)

    def copula_cumulative_density(self, U, V):
        """Computes the cumulative distribution function for the copula, :math:`C(u, v)`

        Args:
            U: `np.ndarray`
            V: `np.ndarray`

        Returns:
            np.array: cumulative probability

        """
        if self.theta < 0:
            raise ValueError("Theta cannot be less or equal than 0 for clayton")

        elif self.theta == 0:
            return np.multiply(U, V)

        elif U == 0 or V == 0:
            return 0

        elif type(U) in (int, float):
            value = np.power(
                np.power(U, -self.theta) + np.power(V, -self.theta) - 1,
                -1.0 / self.theta)

            return value

        else:
            cdfs = [
                np.power(
                    np.power(U[i], -self.theta) + np.power(V[i], -self.theta) - 1,
                    -1.0 / self.theta
                )
                if U[i] > 0 else 0 for i in range(len(U))
            ]

            return np.array([max(x, 0) for x in cdfs])

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative density :math:`C(u|v)^-1`

        Args:
            y: `np.ndarray` value of :math:`C(u|v)`.
            v: `np.ndarray` given value of v.
        """

        if self.theta < 0:
            return V

        else:
            a = np.power(y, self.theta / (-1 - self.theta))
            b = np.power(V, self.theta)
            u = np.power((a + b - 1) / b, -1 / self.theta)
            return u

    def partial_derivative_cumulative_density(self, U, V, y=0):
        """Compute partial derivative :math:`C(u|v)` of cumulative density.

        Args:
            U: `np.ndarray`
            V: `np.ndarray`
            y: `float`

        Returns:
            np.ndarray: Derivatives
        """
        if self.theta == 0:
            return V

        else:
            A = np.power(U, self.theta)
            B = np.power(V, -self.theta) - 1
            h = 1 + np.multiply(A, B)
            h = np.power(h, (-1 - self.theta) / self.theta)
            h = h - y
            return h

    def get_theta(self):
        """Compute theta parameter using Kendall's tau.

        On Clayton copula this is :math:`τ = θ/(θ + 2) \implies θ = 2τ/(1-τ)` with
        :math:`θ ∈ (0, ∞)`.

        On the corner case of :math:`τ = 1`, a big enough number is returned instead of infinity.
        """
        if self.tau == 1:
            theta = 10000

        else:
            theta = 2 * self.tau / (1 - self.tau)

        return theta

    def _sample(self, v, c):
        return self.percent_point(c, v)
