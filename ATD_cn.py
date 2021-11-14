# coding=utf-8
# 作者：BWLL
"""
ATD_cn
======

带有中文注释的算法的主体实现。

Notes
------
元数据 `rcond` :
    由于计算精度问题，取逆将导致小误差被放大，
    因此需要将所有小于 ``rcond`` 的数设为0以避免此问题。\n
    默认值为 :math:`1\\times 10^{-9}`
"""

import warnings
try:
    from typing import Any, Iterable, Optional, Tuple, Union, Callable, final
except ImportError:
    warnings.warn("未能引入类型提示库，可能是Python版本过低。")
try:
    import numpy as np
except ImportError:
    raise ImportWarning("未能引入NumPy，是否未安装？")
    exit(-1)

meta_data = {"trace_update_mode": ["conventional", "emphatic"],
             "w_update_emphasizes": ["complexity", "accuracy"],
             "rcond": 1e-9}  # 元数据
Fraction = Union[float, int]


def learn_func_wrapper(
        func: Callable[[Any, np.ndarray, np.ndarray, float, float, int], Any]
) -> Callable[[Any, np.ndarray, np.ndarray, float, float, int], Any]:
    """
    学习函数的装饰器，用于辅助检查输入数据。
    """
    if not callable(func):
        raise ValueError("错误的装饰器用法或输入不是可调用的函数。")

    def _learn_func(
            self,
            observation: np.ndarray,
            next_observation: np.ndarray,
            reward: float,
            discount: float,
            t: int
    ) -> Any:
        assert observation.shape == (
            self.observation_space_n,), f"当前局面观测数据的形状不正确。应为({self.observation_space_n},)，而不是{observation.shape}"
        assert next_observation.shape == (
            self.observation_space_n,), f"下一个局面观测数据的形状不正确。应为({self.observation_space_n},)，而不是{next_observation.shape}"
        if not (isinstance(reward, Fraction.__args__) and isinstance(discount, Fraction.__args__) and isinstance(
                t, int)):
            raise TypeError("参数类型不正确！")
        if not (t >= 0 and 0 <= discount <= 1):
            raise ValueError("无效的参数！")

        return func(self, observation, next_observation, reward, discount, t)

    return _learn_func


class AbstractAgent:
    """
    AbstractAgent
    ======

    抽象的智能体类，提供基本的一些功能。

    Parameters
    ------
    observation_space_n :
        观测空间大小
    action_space_n :
        动作空间大小
    lambd :
        资格迹所需的λ值
    trace_update_mode :
        资格迹更新方式，取值 ``conventional | emphatic`` 。默认为传统方式( ``conventional`` )
    """

    def __init__(self, observation_space_n: int, action_space_n: int, lambd: Optional[Fraction] = 0,
                 trace_update_mode: Optional[str] = "conventional") -> None:
        if not (isinstance(observation_space_n, int)
                and isinstance(action_space_n, int)
                and isinstance(lambd, Fraction.__args__)
                and isinstance(meta_data["rcond"], Fraction.__args__)):
            raise TypeError("参数类型不正确！")

        self.observation_space_n = observation_space_n
        self.action_space_n = action_space_n
        self.lambd = lambd
        self.trace_update_mode = trace_update_mode

        self.reinit()
        self.reset()

    def reinit(self) -> None:
        """
        让智能体忘记学到的东西。
        """
        self.w = np.empty(self.observation_space_n)  # 任意地初始化权重

    def reset(self) -> None:
        """
        重置与一局游戏有关的智能体参数。应在一局新游戏开始时调用。
        """
        self.F = 0
        self.M = 0
        self.e = np.zeros(self.observation_space_n)

    @learn_func_wrapper
    def learn(
            self,
            observation: np.ndarray,
            next_observation: np.ndarray,
            reward: Fraction,
            discount: Fraction,
            t: int
    ) -> Any:
        """
        训练智能体。这个函数应当装配 `learn_func_wrapper` 装饰器检查输入。

        Parameters
        ------
        observation :
            当前局面
        next_observation :
            下一个局面
        reward :
            局面奖赏
        discount :
            γ折扣，例如除了游戏结束时取0以外全取0.99
        t :
            游戏已进行的步数，从0开始

        Returns
        ------
        Any :
            误差

        Raises
        ------
        NotImplementedError
            相应的学习算法未被实现
        AssertionError
            输入的形状不正确
        TypeError
            参数类型不正确
        ValueError
            参数无效
        """
        raise NotImplementedError("智能体不可训练。")

    def decide(self, next_observations: Iterable[np.ndarray]) -> int:
        """
        让智能体决策一步。

        Parameters
        ------
        next_observations :
            各个动作执行后下一个局面构成的列表

        Returns
        ------
        action : int
            智能体决定执行的动作序号

        Raises
        ------
        ValueError
            意外的错误
        """
        try:
            next_v = [self.w @ next_observation
                      for next_observation in next_observations]
        except ValueError:
            print("发生错误，或许是输入的数据不正确？")
            return -1

        return np.argmax(next_v)

    @final
    def trace_update(self, e: Optional[np.ndarray], observation: np.ndarray, discount: Fraction,
                     lambd: Optional[Fraction], **kwargs) -> np.ndarray:
        """
        资格迹更新（累积迹）。

        Parameters
        ------
        e :
            上一个资格迹。省略即是智能体内存储的结果
        observation :
            当前局面
        discount :
            γ折扣，例如除了游戏结束时取0以外全取0.99
        lambd :
            资格迹所需的λ值。省略即是智能体内存储的结果
        rho :
            仅在使用强调资格迹更新时需要。异策略时，目标策略π与行动策略b选取对应动作概率之比，同策略时为1
        i :
            仅在使用强调资格迹更新时需要。对当前局面的感兴趣程度，均匀感兴趣时可全部取1

        Returns
        ------
        np.ndarray
            新的资格迹

        Raises
        ------
        AssertionError
            输入的形状不正确
        TypeError
            参数类型不正确
        ValueError
            γ折扣无效
        """
        assert observation.shape == (
            self.observation_space_n,), f"当前局面观测数据的形状不正确。应为({self.observation_space_n},)，而不是{observation.shape}"
        if not (isinstance(discount, Fraction.__args__) and isinstance(lambd, Fraction.__args__)):
            raise TypeError("参数类型不正确！")
        if not 0 <= discount <= 1:
            raise ValueError("无效的γ折扣！")
        if e is None:
            e = self.e
        if lambd is None:
            lambd = self.lambd
        if self.trace_update_mode not in meta_data["trace_update_mode"]:
            warnings.warn(
                f"不支持的资格迹更新方式{self.trace_update_mode}！将改为conventional")
            self.trace_update_mode = "conventional"

        return self.__trace_update(e, observation, discount, lambd, **kwargs)

    def __trace_update(self, e: Optional[np.ndarray], observation: np.ndarray, discount: Fraction,
                       lambd: Optional[Fraction], **kwargs) -> np.ndarray:
        """
        内部函数。用于实现具体的资格迹更新算法。
        """
        if self.trace_update_mode == "conventional":
            return discount * lambd * e + observation
        elif self.trace_update_mode == "emphatic":
            return self.__emphatic_trace_update(e, observation, discount, lambd, **kwargs)

    def __emphatic_trace_update(self, e: Optional[np.ndarray], observation: np.ndarray, discount: Fraction,
                                lambd: Optional[Fraction], rho: Optional[Fraction] = 1.,
                                i: Optional[Fraction] = 1.) -> np.ndarray:
        """
        内部函数。用于实现具体的强调资格迹更新算法。
        """
        if not (isinstance(rho, Fraction.__args__) and isinstance(i, Fraction.__args__)):
            raise TypeError("参数类型不正确！")

        self.F = rho * discount * self.F + i
        self.M = lambd * i + (1 - lambd) * self.F

        return rho * (discount * lambd * e + self.M * observation)


class TDAgent(AbstractAgent):
    """
    TDAgent
    ======

    经典时序差分算法。

    Parameters
    ------
    lr :
        学习率
    """

    def __init__(self, lr: Fraction, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(lr, Fraction.__args__):
            raise TypeError("参数类型不正确！")

        self.lr = lr

    @learn_func_wrapper
    def learn(
            self,
            observation: np.ndarray,
            next_observation: np.ndarray,
            reward: Fraction,
            discount: Fraction,
            t: int
    ) -> Any:
        self.e = self.trace_update(
            self.e, observation, discount, self.lambd)  # 更新资格迹
        delta = reward + discount * self.w @ next_observation - self.w @ observation  # 计算时序差分误差
        self.w += self.lr * delta * self.e  # 更新权重

        return delta


class PlainATDAgent(AbstractAgent):
    """
    PlainATDAgent
    ======

    直白的加速的时序差分算法(ATD)。

    Parameters
    ------
    eta :
        半梯度时序差分(TD)学习率
    alpha :
        半梯度均方投影贝尔曼误差(MSPBE)学习率
    """

    def __init__(self, eta: Fraction, alpha: Optional[Fraction] = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        if not (isinstance(eta, Fraction.__args__) and isinstance(alpha, Fraction.__args__)):
            raise TypeError("参数类型不正确！")

        self.eta = eta
        self.alpha = alpha

    def reinit(self) -> None:
        super(PlainATDAgent, self).reinit()
        self.A = np.zeros((self.observation_space_n, self.observation_space_n))

    @learn_func_wrapper
    def learn(
            self,
            observation: np.ndarray,
            next_observation: np.ndarray,
            reward: Fraction,
            discount: Fraction,
            t: int
    ) -> Any:
        beta = 1 / (t + 1)  # 因为这个量要频繁地用到，所以定义成β
        delta = reward + discount * self.w @ next_observation - self.w @ observation  # 计算时序差分误差
        self.e = self.trace_update(
            self.e, observation, discount, self.lambd)  # 更新资格迹

        # 求出A矩阵。A矩阵应是期望值，为了减少计算量，采取渐进式的更新方法
        self.A = (1 - beta) * self.A + beta * self.e.reshape((self.observation_space_n, 1)) \
                 @ (observation - discount * next_observation).reshape((1, self.observation_space_n))

        self.w += (self.alpha * beta * np.linalg.pinv(self.A, rcond=meta_data["rcond"]) + self.eta *
                   np.eye(self.observation_space_n)) @ (delta * self.e)  # 按照论文中的式子更新权重
        # 原始式使用的是1/(1+t)，这里换成了beta

        return delta


class SVDATDAgent(AbstractAgent):
    """
    SVDATDAgent
    ======

    基于奇异值分解(SVD)加速的时序差分算法(ATD)。

    Parameters
    ------
    eta :
        半梯度时序差分(TD)学习率
    alpha :
        半梯度均方投影贝尔曼误差(MSPBE)学习率
    """

    def __init__(self, eta: Fraction, alpha: Optional[Fraction] = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        if not (isinstance(eta, Fraction.__args__) and isinstance(alpha, Fraction.__args__)):
            raise TypeError("参数类型不正确！")

        self.eta = eta
        self.alpha = alpha

    def reinit(self) -> None:
        super(SVDATDAgent, self).reinit()
        self.U, self.V, self.Sigma = np.empty(
            (self.observation_space_n, 0)), np.empty((self.observation_space_n, 0)), np.empty((0, 0))

    def extendWith000(self, mat: np.ndarray) -> np.ndarray:
        """
        用于在二维张量mat周围补0
        """
        return np.pad(mat, ((0, 1), (0, 1)))

    def extendWith010(self, mat: np.ndarray) -> np.ndarray:
        """
        在补0的基础上将右下角设为1
        """
        mat_ = self.extendWith000(mat)
        mat_[-1, -1] = 1
        return mat_

    def svd_update(
            self,
            U: np.ndarray,
            Sigma: np.ndarray,
            V: np.ndarray,
            z: np.ndarray,
            d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        奇异值分解(SVD)更新。它的效果近似等于
        :math:`\\mathbf{U}' \\mathbf{\\Sigma} '\\mathbf{V'}^\\top =
        \\mathbf{U}\\mathbf{\\Sigma}\\mathbf{V}^\\top + \\mathbf{z}\\mathbf{d}^\\top`

        Parameters
        ------
        U :
            矩阵U
        Sigma :
            矩阵∑
        V :
            矩阵V
        z :
            向量z
        d :
            向量d
        epsilon :
            一个很小的数，用于防止除以零

        Returns
        ------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            新的U'、∑'、V'

        Raises
        ------
        TypeError
            类型错误
        ValueError
            数据形状无法相乘
        """
        try:
            U, Sigma, V, z, d = np.asarray(U), np.asarray(
                Sigma), np.asarray(V), np.asarray(z), np.asarray(d)
        except TypeError:
            warnings.warn("不支持的类型！")
            return U, Sigma, V
        if U.ndim != 2 \
                or Sigma.ndim != 2 \
                or V.ndim != 2 \
                or U.shape[1] != Sigma.shape[0] \
                or V.shape[1] != Sigma.shape[1] \
                or U.shape[0] != z.shape[0] \
                or V.shape[0] != d.shape[0]:
            raise ValueError("无法处理的输入！")

        m = U.T @ z
        p = z - U @ m
        n = V.T @ d
        q = d - V @ n

        p_l2 = np.linalg.norm(p)
        q_l2 = np.linalg.norm(q)

        K = self.extendWith000(Sigma) + np.vstack((m, p_l2)
                                                  ) @ np.vstack((n, q_l2)).T

        p = p / p_l2 if p_l2 > 0 else np.zeros_like(p)
        q = q / q_l2 if q_l2 > 0 else np.zeros_like(q)
        U = np.hstack((U, p))
        V = np.hstack((V, q))

        return U, K, V

    @learn_func_wrapper
    def learn(
            self,
            observation: np.ndarray,
            next_observation: np.ndarray,
            reward: Fraction,
            discount: Fraction,
            t: int
    ) -> Any:
        beta = 1 / (t + 1)
        delta = reward + discount * self.w @ next_observation - self.w @ observation
        self.e = self.trace_update(self.e, observation, discount, self.lambd)

        self.U, self.Sigma, self.V = \
            self.svd_update(
                self.U,
                (1 - beta) * self.Sigma,
                self.V,
                np.sqrt(beta) * self.e.reshape((self.observation_space_n, 1)),
                np.sqrt(beta) * (observation - discount *
                                 next_observation).reshape((self.observation_space_n, 1))
            )  # 使用奇异值更新代替直接更新A来降低计算复杂度，提高性能

        self.w += (self.alpha * beta *
                   np.linalg.pinv(self.U @ self.Sigma @ self.V.transpose(), rcond=meta_data["rcond"]) +
                   self.eta *
                   np.eye(self.observation_space_n)) @ (delta * self.e)

        return delta


class DiagonalizedSVDATDAgent(SVDATDAgent):
    """
    DiagonalizedSVDATDAgent
    ======

    将矩阵 :math:`\\mathbf{\Sigma}` 对角化分解的基于奇异值分解(SVD)加速的时序差分算法(ATD)。

    Parameters
    ------
    k :
        最大允许的矩阵大小(k*k)
    svd_diagonalizing :
        决定是否使用奇异值分解来对角化矩阵。默认为 `False`
    w_update_emphasizes :
        权重更新时更注重哪个。可选值：``accuracy(精确度) | complexity(复杂度)``
    """

    def __init__(self, k: int,
                 svd_diagonalizing: Optional[bool] = False,
                 w_update_emphasizes: Optional[str] = "accuracy", **kwargs) -> None:
        super().__init__(**kwargs)
        if not (isinstance(k, int) and isinstance(svd_diagonalizing, bool)):
            raise TypeError("参数类型不正确！")

        self.k = k
        self.svd_diagonalizing = svd_diagonalizing
        self.w_update_emphasizes = w_update_emphasizes

    def reinit(self) -> None:
        super(DiagonalizedSVDATDAgent, self).reinit()
        self.L, self.R = np.empty((0, 0)), np.empty((0, 0))

    def svd_update(
            self,
            U: np.ndarray,
            Sigma: np.ndarray,
            V: np.ndarray,
            z: np.ndarray,
            d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            U, Sigma, V, z, d = np.asarray(U), np.asarray(
                Sigma), np.asarray(V), np.asarray(z), np.asarray(d)
        except TypeError:
            warnings.warn("不支持的类型！")
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
            raise ValueError("无法处理的输入！")

        m = self.L.T @ (U.T @ z)
        p = z - U @ (self.L @ m)
        n = self.R.T @ (V.T @ d)
        q = d - V @ (self.R @ n)

        p_l2 = np.linalg.norm(p)
        q_l2 = np.linalg.norm(q)

        K = self.extendWith000(Sigma) + np.vstack((m, p_l2)
                                                  ) @ np.vstack((n, q_l2)).T

        if self.svd_diagonalizing:
            L_, Sigma, R_ = np.linalg.svd(K)
            Sigma = np.diagflat(Sigma)
            R_ = R_.T
        else:
            L_, Sigma, R_ = self.diagonalize(K)

        self.L = self.extendWith010(self.L) @ L_
        self.R = self.extendWith010(self.R) @ R_
        p = p / p_l2 if p_l2 > meta_data["rcond"] else np.zeros_like(p)  # 向量很小时取零向量，因为零向量不会影响伪逆。
        q = q / q_l2 if q_l2 > meta_data["rcond"] else np.zeros_like(q)
        U = np.hstack((U, p))
        V = np.hstack((V, q))

        if self.L.shape[1] >= 2 * self.k:
            Sigma = Sigma[:self.k, :self.k]
            U = U @ self.L
            U = U[:, :self.k]
            V = V @ self.R
            V = V[:, :self.k]
            self.L, self.R = np.eye(self.k), np.eye(self.k)

        return U, Sigma, V

    @staticmethod
    def diagonalize(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将矩阵 :math:`\\mathbf{K}` 对角化，且是完全正交的。

        Parameters
        ------
        K : np.ndarray
            待被对角化的矩阵

        Returns
        ------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            对角化完毕的三个矩阵

        Raises
        ------
        ValueError
            矩阵的形状不正确
        TypeError
            不支持的输入类型
        """
        try:
            K = np.asarray(K)
        except TypeError:
            raise TypeError("不支持的类型！")
        if K.shape[0] != K.shape[1]:
            raise ValueError("不支持非方阵的对角化操作。")

        r, l, alpha, beta = [], [], [], []
        # 任取一个单位向量
        unit = np.full((K.shape[0], 1), 1 / np.sqrt(K.shape[0]))
        r.append(unit)

        for j in range(K.shape[0]):
            l.append(K @ r[j])
            for i in range(j):
                l[j] -= (l[i].T @ l[j]) * l[i]
            alpha.append(np.linalg.norm(l[j]))
            l[j] = l[j] / alpha[j] if alpha[j] > meta_data["rcond"] \
                else np.zeros_like(l[j])  # 很小的向量基本都是由于误差引起，故直接置零。取零向量而不取其他向量的原因见上。下同。

            r.append(K.T @ l[j])
            for i in range(j + 1):
                r[j + 1] -= (r[i].T @ r[j + 1]) * r[i]
            beta.append(np.linalg.norm(r[j + 1]))
            r[j + 1] = r[j + 1] / beta[j] if beta[j] > meta_data["rcond"] \
                else np.zeros_like(r[j + 1])

        L2, Sigma, R2 = np.linalg.svd(
            np.diagflat(alpha) + np.diagflat(beta[:-1], 1))  # 通过α和β构造双对角矩阵再奇异值分解
        L1, R1 = np.array(l)[..., 0].T, np.array(r[:-1])[..., 0].T
        return L1 @ L2, np.diagflat(Sigma), R1 @ R2.T

    @learn_func_wrapper
    def learn(
            self,
            observation: np.ndarray,
            next_observation: np.ndarray,
            reward: Fraction,
            discount: Fraction,
            t: int
    ) -> Any:
        if self.w_update_emphasizes not in meta_data["w_update_emphasizes"]:
            warnings.warn(
                f"意外的权重更新方式{self.w_update_emphasizes}！将改为accuracy")
            self.w_update_emphasizes = "accuracy"

        beta = 1 / (t + 1)
        delta = reward + discount * self.w @ next_observation - self.w @ observation
        self.e = self.trace_update(self.e, observation, discount, self.lambd)

        self.U, self.Sigma, self.V = \
            self.svd_update(
                self.U,
                (1 - beta) * self.Sigma,
                self.V,
                np.sqrt(beta) * self.e.reshape((self.observation_space_n, 1)),
                np.sqrt(beta) * (observation - discount *
                                 next_observation).reshape((self.observation_space_n, 1))
            )  # 使用奇异值更新代替直接更新A来降低计算复杂度，提高性能

        # 参考论文降低了复杂度。
        if self.w_update_emphasizes == "accuracy":
            # 原本直接按公式更新：
            self.w += (self.alpha * beta *
                       np.linalg.pinv(self.U @ self.L @ self.Sigma @ (self.V @ self.R).T, rcond=meta_data["rcond"]) +
                       self.eta *
                       np.eye(self.observation_space_n)) @ (delta * self.e)
        elif self.w_update_emphasizes == "complexity":
            # 降低复杂度的更新方法：
            self.w += self.alpha * beta * self.V @ self.R @ (np.diagflat(
                [(1 / sigma if abs(sigma) > meta_data["rcond"] else 0) for sigma in np.diagonal(self.Sigma)]
            ) @ (self.L.T @ (self.U.T @ (delta * self.e)))) + self.eta * delta * self.e

        return delta


def _svd_minibatch_update(
        U: np.ndarray,
        Sigma: np.ndarray,
        V: np.ndarray,
        Z: np.ndarray,
        D: np.ndarray, r: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    批量奇异值更新，以备不时之需。
    """

    Q_Z, R_Z = np.linalg.qr((1 - U @ U.transpose()) @ Z)
    Q_D, R_D = np.linalg.qr((1 - V @ V.transpose()) @ D)

    K = np.pad(Sigma, ((0, 1), (0, 1))) + np.vstack((U.transpose() @ Z, R_Z)) @ np.vstack((V.transpose()
                                                                                           @ D, R_D)).transpose()

    L, Sigma_diagonalized, R = np.linalg.svd(K)
    Sigma = np.diag(Sigma_diagonalized)

    U = np.hstack((U, Q_Z)) @ L
    V = np.hstack((V, Q_D)) @ R

    return U, Sigma, V


print(
    """
    ATD算法已成功载入。
    """.strip()
)

if __name__ == "__main__":
    print(
        """
        这是ATD算法的实现类，无法直接运行。请另行编写程序调用。
        """.strip()
    )
