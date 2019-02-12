import numpy as np

from copulas.bivariate.base import Bivariate, CopulaTypes


class Clayton(Bivariate):
    """Class for clayton copula model."""

    copula_type = CopulaTypes.CLAYTON
    theta_interval = [-1, float('inf')]
    invalid_thetas = [0]

    def generator(self, t):
        """Return the generator function."""
        self.check_fit()

        return (1.0 / self.theta) * (np.power(t, -self.theta) - 1)

    def probability_density(self, X):
        """Compute probability density function for given copula family.

        Args:
            X: `np.ndarray`

        Returns:
            np.array: Probability density for the input values.
        """
        self.check_fit()

        U, V = self.split_matrix(X)

        a = (self.theta + 1) * np.power(np.multiply(U, V), -(self.theta + 1))
        b = np.power(U, -self.theta) + np.power(V, -self.theta) - 1
        c = -(2 * self.theta + 1) / self.theta
        return a * np.power(b, c)

    def cumulative_distribution(self, X):
        """Computes the cumulative distribution function for the copula, :math:`C(u, v)`

        Args:
            X: `np.ndarray`

        Returns:
            np.array: cumulative probability
        """
        self.check_fit()

        U, V = self.split_matrix(X)

        if (V == 0).all() or (U == 0).all():
            return np.zeros(V.shape[0])

        else:
            cdfs = [
                np.power(
                    np.power(U[i], -self.theta) + np.power(V[i], -self.theta) - 1,
                    -1.0 / self.theta
                )
                if U[i] > 0 else 0
                for i in range(len(U))
            ]

            return np.array([max(x, 0) for x in cdfs])

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`C(u|v)^-1`

        Args:
            y: `np.ndarray` value of :math:`C(u|v)`.
            v: `np.ndarray` given value of v.
        """
        self.check_fit()

        if self.theta < 0:
            return V

        else:
            a = np.power(y, self.theta / (-1 - self.theta))
            b = np.power(V, self.theta)
            u = np.power((a + b - 1) / b, -1 / self.theta)
            return u

    def partial_derivative(self, X, y=0):
        """Compute partial derivative :math:`C(u|v)` of cumulative distribution.

        Args:
            X: `np.ndarray`
            y: `float`

        Returns:
            np.ndarray: Derivatives
        """
        self.check_fit()

        U, V = self.split_matrix(X)

        if self.theta == 0:
            return V

        else:
            A = np.power(V, -self.theta - 1)
            B = np.power(V, -self.theta) + np.power(U, -self.theta) - 1
            h = np.power(B, (-1 - self.theta) / self.theta)
            return np.multiply(A, h) - y

    def compute_theta(self):
        """Compute theta parameter using Kendall's tau.

        On Clayton copula this is :math:`τ = θ/(θ + 2) \\implies θ = 2τ/(1-τ)` with
        :math:`θ ∈ (0, ∞)`.

        On the corner case of :math:`τ = 1`, a big enough number is returned instead of infinity.
        """
        if self.tau == 1:
            theta = 10000

        else:
            theta = 2 * self.tau / (1 - self.tau)

        return theta
