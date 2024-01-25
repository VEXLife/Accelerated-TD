#!python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Midden Vexu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# @Author：Midden Vexu
# Original repository is https://github.com/VEXLife/Accelerated-TD
# Reference: https://arxiv.org/pdf/1611.09328.pdf

"""
atd
======

Now you're able to switch between backends including NumPy and PyTorch(CPU) via
setting environment variable "ATD_BACKEND".\n
For more details, see `README.md`.

Notes
------
Meta data `rcond` :
    The universal ``rcond`` parameter for all ``numpy.pinv``.\n
    For more details, see https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv.\n
    Frankly speaking, this parameter stands for a cutoff for small singular values.
    Default is :math:`1\\times 10^{-5}`
"""
import sys
import warnings
from math import sqrt

if sys.version_info < (3, 9):
    warnings.warn("You're suggested to upgrade your Python interpreter.", category=ImportWarning)

try:
    from typing import Any, Iterable, Optional, Tuple, Union, Callable, Final, final
    from abc import abstractmethod
    from functools import wraps
except ImportError:
    warnings.warn("Unable to import type hinting library, "
                  "possibly because you have to upgrade your Python interpreter.", category=ImportWarning)


    def original_decorator(obj: Callable) -> Callable:
        return obj


    abstractmethod = final = wraps = original_decorator
    Any = Iterable = Optional = Tuple = Union = Callable = Final = None

try:
    if sys.version_info < (3, 10):
        # Support for old version
        from backend_manager_39 import Backend, Matrix, Decimal, isinstance, extend_with_000, extend_with_010
    else:
        from backend_manager_310 import Backend, Matrix, Decimal, extend_with_000, extend_with_010
except ImportError:
    raise ImportError("Unable to import the specified backend!")
    exit(-1)

meta_data: dict = {"trace_update_mode": {},
                   "w_update_emphasizes": ["complexity", "accuracy"],
                   "rcond": 1e-5}  # Meta data
TraceUpdateFunction: Final = Callable[[Any, Matrix, Decimal, Optional[Matrix],
                                       Optional[Decimal], Optional[Decimal],
                                       Optional[Decimal]], Matrix]


def learn_func_wrapper(
        func: Callable[[Any, Matrix, Matrix, float, float, int], Any]
) -> Callable[[Any, Matrix, Matrix, float, float, int], Any]:
    """
    The decorator for the learn function. Helpful for checking input.
    """
    if not callable(func):
        raise ValueError("Unexpected decorator usage.")

    @wraps(func)
    def _learn_func(
            self: AbstractAgent,
            observation: Matrix,
            next_observation: Matrix,
            reward: float,
            discount: float,
            t: int
    ) -> Any:
        assert observation.shape == (
            self.observation_space_n,), f"Bad observation shape. Expected ({self.observation_space_n},), not {observation.shape}"
        assert next_observation.shape == (
            self.observation_space_n,), f"Bad next observation shape. Expected ({self.observation_space_n},), not {next_observation.shape}"
        if not (isinstance(reward, Decimal) and isinstance(discount, Decimal)
                and isinstance(t, int) and isinstance(self, AbstractAgent)):
            raise TypeError("Invalid input type!")
        if not (t >= 0 and 0 <= discount <= 1):
            raise ValueError("Invalid hyperparameter!")

        self.lr = self.lr_func(t)  # Calculate the new learning rate

        return func(self, observation, next_observation, reward, discount, t)

    return _learn_func


def register_trace_update_func(
        mode_name: str
) -> Callable[[TraceUpdateFunction], TraceUpdateFunction]:
    """
    Decorator for registering trace update functions.
    """

    def _trace_update_func_wrapper(
            func: TraceUpdateFunction
    ) -> TraceUpdateFunction:
        """
        Decorator for trace update functions. Helpful for checking input.
        """

        if not callable(func):
            raise ValueError("Unexpected decorator usage.")
        if not isinstance(mode_name, str):
            raise TypeError("Invalid trace update mode type.")

        @wraps(func)
        def _trace_update_func(self: Any, observation: Matrix,
                               discount: Decimal, e: Optional[Matrix] = None,
                               lambd: Optional[Decimal] = None, rho: Optional[Decimal] = 1.,
                               i: Optional[Decimal] = 1.) -> Matrix:
            assert observation.shape == (
                self.observation_space_n,), f"Bad observation shape. Expected ({self.observation_space_n},), not {observation.shape}"
            if not (isinstance(discount, Decimal) and isinstance(lambd, Decimal)
                    and isinstance(self, AbstractAgent)):
                raise TypeError("Invalid input type!")
            if not 0 <= discount <= 1:
                raise ValueError("Invalid discount parameter!")
            if e is None:
                e = self.e
            if lambd is None:
                lambd = self.lambd

            return func(self=self, observation=observation, discount=discount, e=e, lambd=lambd, rho=rho, i=i)

        meta_data["trace_update_mode"][mode_name] = _trace_update_func
        return _trace_update_func

    return _trace_update_func_wrapper


class AbstractAgent:
    """
    AbstractAgent
    ======

    The abstract agent class, offering some fundamental functions.

    Parameters
    ------
    observation_space_n :
        The shape(1-D) of observation space
    action_space_n :
        The shape(1-D) of action space
    lr :
        learning rate, could be a function with time step as input and learning rate as output, or a float representing
        constant learning rate
    lambd :
        λ for trace updating
    trace_update_mode :
        Trace update mode, should be ``conventional | emphatic`` . Default is ``conventional``.

    Raises
    ------
    TypeError
        Invalid input type
    AssertionError
        Unable to deal with the learning rate input
    """

    def __init__(self, observation_space_n: int, action_space_n: int,
                 lr: Union[Callable[[int], Decimal], Decimal], lambd: Optional[Decimal] = 0,
                 trace_update_mode: Optional[str] = "conventional") -> None:
        if not (isinstance(observation_space_n, int)
                and isinstance(action_space_n, int)
                and isinstance(lambd, Decimal)
                and isinstance(meta_data["rcond"], Decimal)
                and isinstance(trace_update_mode, str)):
            raise TypeError("Invalid input type!")
        if trace_update_mode not in meta_data["trace_update_mode"].keys():
            warnings.warn(
                f"Not supported trace update mode: {trace_update_mode}! Will be set to conventional.")
            trace_update_mode = "conventional"
        if isinstance(lr, Decimal):
            self.lr_func = lambda t: lr
        else:
            assert callable(lr), "Unable to deal with the learning rate input."
            self.lr_func = lr

        self.observation_space_n = observation_space_n
        self.action_space_n = action_space_n
        self.lambd = lambd
        self.trace_update = meta_data["trace_update_mode"][trace_update_mode]  # type: TraceUpdateFunction

        self.reinit()
        self.reset()

    def reinit(self) -> None:
        """
        Make the agent forget what it learned.
        """
        self.w = Backend.empty(self.observation_space_n)  # Initialize the weight arbitrarily

    def reset(self) -> None:
        """
        Reset everything of the agent. Should be invoked when a game begins.
        """
        self.F = 0
        self.M = 0
        self.e = Backend.zeros(self.observation_space_n)

    @abstractmethod
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:
        """
        Train the agent. Should be decorated with ``@learn_func_wrapper`` .

        Parameters
        ------
        observation :
            Current observation
        next_observation :
            Next observation
        reward :
            Reward
        discount :
            γ discount. 0 for the terminal step and 0.99 for the rest for example
        t :
            Time step. Starts from 0

        Returns
        ------
        Any :
            The loss

        Raises
        ------
        NotImplementedError
            This learn function has not been implemented yet
        AssertionError
            Invalid input shape
        TypeError
            Invalid input type
        ValueError
            Invalid hyperparameter
        """
        raise NotImplementedError("The agent is not trainable!")

    def decide(self, next_observations: Iterable[Matrix]) -> int:
        """
        Ask the agent to pick an action.

        Parameters
        ------
        next_observations :
            A list consisted of all the next observations

        Returns
        ------
        action : int
            The action index picked by the agent

        Raises
        ------
        ValueError
            Unexpected error
        """
        warnings.simplefilter("default", DeprecationWarning)
        warnings.warn("This function has not been tested yet!", category=DeprecationWarning)

        try:
            next_v = [self.w @ next_observation
                      for next_observation in next_observations]
        except ValueError:
            print("Unexcepted error, maybe the input shape is invalid?")
            return -1

        return Backend.argmax(next_v)

    @staticmethod
    @final
    def trace_update(self, observation: Matrix, discount: Decimal, e: Optional[Matrix] = None,
                     lambd: Optional[Decimal] = None, rho: Optional[Decimal] = 1.,
                     i: Optional[Decimal] = 1.) -> Matrix:
        """
        Trace update function (accumulative).\n
        If you're about to include your own trace update function, please do not override this function, but define
        a new function instead, with ``@staticmethod`` and
        ``@register_trace_update_func("<Your trace update function name>")`` decorators.

        Parameters
        ------
        self :
            The agent object for trace update
        observation :
            Current observation
        discount :
            γ discount. 0 for the terminal step and 0.99 for the rest for example
        e :
            Previous trace. Omit it to use the one stored in the agent
        lambd :
            λ for trace updating. Omit it to use the one stored in the agent
        rho :
            Only needed when emphatic trace update is required.
            In the off-policy context, it is the quotient of the probability to choose the action if applied the target
            policy π and the probability if applied the behaviour policy b, namely :math:`\\frac{π(a)}{b(a)}` .
            In the on-policy context, it should be 1.
        i :
            Only needed when emphatic trace update is required.
            How much is the agent interested in the current observation. If averagely interested, then it is 1.

        Returns
        ------
        Matrix
            New trace

        Raises
        ------
        AssertionError
            Invalid input shape
        TypeError
            Invalid input type
        ValueError
            Invalid γ discount
        """
        ...

    @staticmethod
    @register_trace_update_func("conventional")
    def __trace_update(*, self, observation: Matrix, discount: Decimal, e: Optional[Matrix] = None,
                       lambd: Optional[Decimal] = None, **kwargs) -> Matrix:
        """
        Internal function.
        The implementation of concrete conventional trace update algorithm.
        """
        return discount * lambd * e + observation

    @staticmethod
    @register_trace_update_func("emphatic")
    def __emphatic_trace_update(*, self, observation: Matrix, discount: Decimal, e: Optional[Matrix] = None,
                                lambd: Optional[Decimal] = None, rho: Optional[Decimal] = 1.,
                                i: Optional[Decimal] = 1., **kwargs) -> Matrix:
        """
        Internal function.
        The implementation of concrete emphatic trace update algorithm.
        """
        if not (isinstance(rho, Decimal) and isinstance(i, Decimal)):
            raise TypeError("Invalid input type!")

        self.F = rho * discount * self.F + i
        self.M = lambd * i + (1 - lambd) * self.F

        return rho * (discount * lambd * e + self.M * observation)


class TDAgent(AbstractAgent):
    """
    TDAgent
    ======

    Conventional temporal difference learning algorithm.

    See Also
    ------
    ``TDAgent``
    """

    @learn_func_wrapper
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:
        self.e = self.trace_update(self, observation, discount, self.e, self.lambd)  # Updates the trace
        delta = reward + discount * self.w @ next_observation - self.w @ observation  # Calculate the TD error
        self.w += self.lr * delta * self.e  # Updates the weight

        return delta


class PlainATDAgent(AbstractAgent):
    """
    PlainATDAgent
    ======

    Plain accelerated temporal difference learning algorithm(ATD).

    Parameters
    ------
    eta :
        Learning rate for semi-gradient TD.
    lr :
        Learning rate for semi-gradient mean squared projected Bellman error(MSPBE).
    """

    def __init__(self,
                 eta: Decimal,
                 lr: Optional[Union[Callable[[int], Decimal], Decimal]] = lambda t: 1 / (t + 1),
                 **kwargs) -> None:
        super().__init__(lr=lr, **kwargs)
        if not (isinstance(eta, Decimal)):
            raise TypeError("Invalid input type!")

        self.eta = eta

    def reinit(self) -> None:
        super(PlainATDAgent, self).reinit()
        self.A = Backend.zeros((self.observation_space_n, self.observation_space_n))

    @learn_func_wrapper
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:
        beta = 1 / (t + 1)  # As this value is frequently used, assign it to a variable β
        delta = reward + discount * self.w @ next_observation - self.w @ observation  # Calculates the TD error
        self.e = self.trace_update(self, observation, discount, self.e, self.lambd)  # Updates the trace

        # Calculates the matrix A. A should be the expectation, so use incremental update method to reduce complexity
        self.A = (1 - beta) * self.A + beta * self.e.reshape((self.observation_space_n, 1)) \
                 @ (observation - discount * next_observation).reshape((1, self.observation_space_n))

        self.w += (self.lr * Backend.linalg.pinv(self.A, rcond=meta_data["rcond"]) + self.eta *
                   Backend.eye(self.observation_space_n)) @ (delta * self.e)  # Updates the weight accordingly
        # Originally 1/(1+t) is used, replacing it with beta

        return delta


class SVDATDAgent(AbstractAgent):
    """
    SVDATDAgent
    ======

    The ATD algorithm based on SVD decomposition.

    Parameters
    ------
    eta :
        Learning rate for semi-gradient TD.
    lr :
        Learning rate for semi-gradient mean squared projected Bellman error(MSPBE).

    See Also
    ------
    ``PlainATDAgent``
    """

    def __init__(self,
                 eta: Decimal,
                 lr: Optional[Union[Callable[[int], Decimal], Decimal]] = lambda t: 1 / (t + 1),
                 **kwargs) -> None:
        super().__init__(lr=lr, **kwargs)
        if not (isinstance(eta, Decimal)):
            raise TypeError("Invalid input type!")

        self.eta = eta

    def reinit(self) -> None:
        super(SVDATDAgent, self).reinit()
        self.U, self.V, self.Sigma = Backend.empty(
            (self.observation_space_n, 0)), Backend.empty((self.observation_space_n, 0)), Backend.empty((0, 0))

    def svd_update(
            self,
            U: Matrix,
            Sigma: Matrix,
            V: Matrix,
            z: Matrix,
            d: Matrix
    ) -> Tuple[Matrix, Matrix, Matrix]:
        """
        SVD update. It is the same as
        :math:`\\mathbf{U}' \\mathbf{\\Sigma} '\\mathbf{V'}^\\top =
        \\mathbf{U}\\mathbf{\\Sigma}\\mathbf{V}^\\top + \\mathbf{z}\\mathbf{d}^\\top`

        Parameters
        ------
        U :
            The matrix U
        Sigma :
            The matrix ∑
        V :
            The matrix V
        z :
            The vector z
        d :
            The vector d

        Returns
        ------
        Tuple[Matrix, Matrix, Matrix]
            The new updated U'、∑'、V'

        Raises
        ------
        TypeError
            Wrong input type
        ValueError
            Cannot multiply the matrices.
        """
        try:
            U, Sigma, V, z, d = Backend.convert_to_matrix_func(U), Backend.convert_to_matrix_func(
                Sigma), Backend.convert_to_matrix_func(V), Backend.convert_to_matrix_func(
                z), Backend.convert_to_matrix_func(d)
        except TypeError:
            warnings.warn("Wrong input type!")
            return U, Sigma, V
        if U.ndim != 2 \
                or Sigma.ndim != 2 \
                or V.ndim != 2 \
                or U.shape[1] != Sigma.shape[0] \
                or V.shape[1] != Sigma.shape[1] \
                or U.shape[0] != z.shape[0] \
                or V.shape[0] != d.shape[0]:
            raise ValueError("Unable to handle the input!")

        m = U.T @ z
        p = z - U @ m
        n = V.T @ d
        q = d - V @ n

        p_l2 = Backend.linalg.norm(p)
        q_l2 = Backend.linalg.norm(q)

        K = extend_with_000(Sigma) + Backend.vstack((m, p_l2)
                                                         ) @ Backend.vstack((n, q_l2)).T

        p = p / p_l2 if p_l2 > 0 else Backend.zeros_like(p)
        q = q / q_l2 if q_l2 > 0 else Backend.zeros_like(q)
        U = Backend.hstack((U, p))
        V = Backend.hstack((V, q))

        return U, K, V

    @learn_func_wrapper
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:
        beta = 1 / (t + 1)
        delta = reward + discount * self.w @ next_observation - self.w @ observation
        self.e = self.trace_update(self, observation, discount, self.e, self.lambd)

        self.U, self.Sigma, self.V = \
            self.svd_update(
                self.U,
                (1 - beta) * self.Sigma,
                self.V,
                sqrt(beta) * self.e.reshape((self.observation_space_n, 1)),
                sqrt(beta) * (observation - discount *
                                      next_observation).reshape((self.observation_space_n, 1))
            )  # Uses SVD update to reduce the complexity, enhancing the performance

        self.w += (self.lr *
                   Backend.linalg.pinv(self.U @ self.Sigma @ self.V.T, rcond=meta_data["rcond"]) +
                   self.eta *
                   Backend.eye(self.observation_space_n)) @ (delta * self.e)

        return delta


class DiagonalizedSVDATDAgent(SVDATDAgent):
    """
    DiagonalizedSVDATDAgent
    ======

    Diagonalizing :math:`\\mathbf{\\Sigma}` and SVD decomposition based ATD。

    Parameters
    ------
    k :
        The largest allowed size of matrices(k*k)
    svd_diagonalizing :
        Decides whether to use svd decomposition to diagonalize the matrix with orthogonality. Default is `False`
    w_update_emphasizes :
        Decides which one comes first when updating the weight. Should be one of ``accuracy | complexity``
    """

    def __init__(self, k: int,
                 svd_diagonalizing: Optional[bool] = False,
                 w_update_emphasizes: Optional[str] = "accuracy", **kwargs) -> None:
        super().__init__(**kwargs)
        if not (isinstance(k, int) and isinstance(svd_diagonalizing, bool)):
            raise TypeError("Invalid input type!")

        self.k = k
        self.svd_diagonalizing = svd_diagonalizing
        self.w_update_emphasizes = w_update_emphasizes

    def reinit(self) -> None:
        super(DiagonalizedSVDATDAgent, self).reinit()
        self.L, self.R = Backend.empty((0, 0)), Backend.empty((0, 0))

    def svd_update(
            self,
            U: Matrix,
            Sigma: Matrix,
            V: Matrix,
            z: Matrix,
            d: Matrix
    ) -> Tuple[Matrix, Matrix, Matrix]:
        try:
            U, Sigma, V, z, d = Backend.convert_to_matrix_func(U), Backend.convert_to_matrix_func(
                Sigma), Backend.convert_to_matrix_func(V), Backend.convert_to_matrix_func(
                z), Backend.convert_to_matrix_func(d)
        except TypeError:
            warnings.warn("Wrong input type!")
            return U, Sigma, V
        if U.ndim != 2 \
                or Sigma.ndim != 2 \
                or V.ndim != 2 \
                or self.L.shape[1] != Sigma.shape[0] \
                or self.R.shape[1] != Sigma.shape[1] \
                or self.L.shape[0] != U.shape[1] \
                or self.R.shape[0] != V.shape[1] \
                or U.shape[0] != z.shape[0] \
                or V.shape[0] != d.shape[0]:
            raise ValueError("Unable to handle the input!")

        m = self.L.T @ (U.T @ z)
        p = z - U @ (self.L @ m)
        n = self.R.T @ (V.T @ d)
        q = d - V @ (self.R @ n)

        p_l2 = Backend.linalg.norm(p)
        q_l2 = Backend.linalg.norm(q)

        K = extend_with_000(Sigma) + Backend.vstack((m, p_l2)
                                                         ) @ Backend.vstack((n, q_l2)).T

        if self.svd_diagonalizing:
            L_, Sigma, R_ = Backend.linalg.svd(K)
            Sigma = Backend.diagflat(Sigma)
            R_ = R_.T
        else:
            L_, Sigma, R_ = self.diagonalize(K)

        self.L = extend_with_010(self.L) @ L_
        self.R = extend_with_010(self.R) @ R_
        # Takes zero vector if the vector is infinitesimal, as it doesn't affects the Moore-Penrose inverse
        p = p / p_l2 if p_l2 > meta_data["rcond"] else Backend.zeros_like(p)
        q = q / q_l2 if q_l2 > meta_data["rcond"] else Backend.zeros_like(q)
        U = Backend.hstack((U, p))
        V = Backend.hstack((V, q))

        if self.L.shape[1] >= 2 * self.k:
            Sigma = Sigma[:self.k, :self.k]
            U = U @ self.L
            U = U[:, :self.k]
            V = V @ self.R
            V = V[:, :self.k]
            self.L, self.R = Backend.eye(self.k), Backend.eye(self.k)

        return U, Sigma, V

    @staticmethod
    def diagonalize(K: Matrix) -> Tuple[Matrix, Matrix, Matrix]:
        """
        Diagonalizes :math:`\\mathbf{K}` with orthogonality

        Parameters
        ------
        K : Matrix
            The target matrix

        Returns
        ------
        Tuple[Matrix, Matrix, Matrix]
            New diagonalized matrices

        Raises
        ------
        ValueError
            Cannot multiply the matrices
        TypeError
            Invalid input type
        """
        try:
            K = Backend.convert_to_matrix_func(K)
        except TypeError:
            raise TypeError("Invalid input type!")
        if K.shape[0] != K.shape[1]:
            raise ValueError("Diagonalizing of non-square matrices is not supported!")

        r, l, alpha, beta = [], [], [], []
        # Pick a unit vector arbitrarily
        unit = Backend.full((K.shape[0], 1), 1 / sqrt(K.shape[0]))
        r.append(unit)

        for j in range(K.shape[0]):
            l.append(K @ r[j])
            for i in range(j):
                l[j] -= (l[i].T @ l[j]) * l[i]
            alpha.append(Backend.linalg.norm(l[j]))
            l[j] = l[j] / alpha[j] if alpha[j] > meta_data["rcond"] \
                else Backend.zeros_like(l[j])  # Sets the infinitesimal vectors to zero vectors directly like above.

            r.append(K.T @ l[j])
            for i in range(j + 1):
                r[j + 1] -= (r[i].T @ r[j + 1]) * r[i]
            beta.append(Backend.linalg.norm(r[j + 1]))
            r[j + 1] = r[j + 1] / beta[j] if beta[j] > meta_data["rcond"] \
                else Backend.zeros_like(r[j + 1])  # The same as above.

        # Builds the bi-diagonalized matrix with α and β before decomposition
        L2, Sigma, R2 = Backend.linalg.svd(
            Backend.diagflat(Backend.create_matrix_func(alpha))
            + Backend.diagflat(Backend.create_matrix_func(beta[:-1]), 1))
        L1, R1 = Backend.hstack(l), Backend.hstack(r[:-1])
        return L1 @ L2, Backend.diagflat(Sigma), R1 @ R2.T

    @learn_func_wrapper
    def learn(
            self,
            observation: Matrix,
            next_observation: Matrix,
            reward: Decimal,
            discount: Decimal,
            t: int
    ) -> Any:
        if self.w_update_emphasizes not in meta_data["w_update_emphasizes"]:
            warnings.warn(
                f"Unexpected weight update emphasizes parameter {self.w_update_emphasizes}! Will be set to accuracy.")
            self.w_update_emphasizes = "accuracy"

        beta = 1 / (t + 1)
        delta = reward + discount * self.w @ next_observation - self.w @ observation
        self.e = self.trace_update(self, observation, discount, self.e, self.lambd)

        self.U, self.Sigma, self.V = \
            self.svd_update(
                self.U,
                (1 - beta) * self.Sigma,
                self.V,
                sqrt(beta) * self.e.reshape((self.observation_space_n, 1)),
                sqrt(beta) * (observation - discount *
                                      next_observation).reshape((self.observation_space_n, 1))
            )  # Uses SVD update to reduce the complexity, enhancing the performance

        # Reduces the complexity according to the paper
        if self.w_update_emphasizes == "accuracy":
            # Originally:
            self.w += (self.lr *
                       Backend.linalg.pinv(self.U @ self.L @ self.Sigma @ (self.V @ self.R).T,
                                           rcond=meta_data["rcond"]) +
                       self.eta *
                       Backend.eye(self.observation_space_n)) @ (delta * self.e)
        elif self.w_update_emphasizes == "complexity":
            # The one with less complexity:
            self.w += self.lr * self.V @ self.R @ (Backend.diagflat(
                Backend.create_matrix_func(
                    [(1 / sigma if abs(sigma) > meta_data["rcond"] else 0) for sigma in Backend.diagonal(self.Sigma)]
                )
            ) @ (self.L.T @ (self.U.T @ (delta * self.e)))) + self.eta * delta * self.e

        return delta


def _svd_minibatch_update(
        U: Matrix,
        Sigma: Matrix,
        V: Matrix,
        Z: Matrix,
        D: Matrix, r: int
) -> Tuple[Matrix, Matrix, Matrix]:
    """
    A backup function.
    """

    Q_Z, R_Z = Backend.linalg.qr((1 - U @ U.transpose()) @ Z)
    Q_D, R_D = Backend.linalg.qr((1 - V @ V.transpose()) @ D)

    K = Backend.pad(Sigma, ((0, 1), (0, 1))) + Backend.vstack((U.transpose() @ Z, R_Z)) @ Backend.vstack((V.transpose()
                                                                                                          @ D,
                                                                                                          R_D)).transpose()

    L, Sigma_diagonalized, R = Backend.linalg.svd(K)
    Sigma = Backend.diag(Sigma_diagonalized)

    U = Backend.hstack((U, Q_Z)) @ L
    V = Backend.hstack((V, Q_D)) @ R

    return U, Sigma, V


print(
    """
    ATD algorithm module has been ready.
    """.strip()
)

if __name__ == "__main__":
    print(
        """
        This is an implementation of ATD, please invoke it instead of running it directly.
        """.strip()
    )
