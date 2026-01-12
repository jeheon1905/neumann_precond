import torch
from itertools import product, combinations_with_replacement
from gospel.util import timer


class Mixing:
    def __init__(
        self, what="density", scheme="pulay", history=10, parameter=0.2, pulay_beta=0.3
    ):
        """Mix densities or potentials to accelerate SCF convergence.

        :type  what: str, optional
        :param what:
            what to mix, options=['density', 'potential'], defaults to 'density'
        :type  scheme: str, optional
        :param scheme:
            mixing scheme, options=['linear', 'pulay', None], defaults to 'pulay'
        :type  history: int, optional
        :param history:
            the maximum number of densities or potentials for making new mixed one.
            It's needed for pulay mixing, defaults to 10
        :type  parameter: float, optional
        :param parameter:
            A parameter for how much density or potential will be updated.
            It's needed for linear mixing, defaults to 0.2
        :type  pulay_beta: float, optional
        :param pulay_beta:
            A parameter for how much residue will be added.
            It's needed for pulay mixing, defaults to 0.3

        **Example**

        >>> mixing = Mixing()
        >>> mixing.append(old)
        >>> new = mixing.get_mixed_one()
        """

        self.__what = what
        self.__scheme = scheme
        self.__history = history
        self.__parameter = parameter
        self.__pulay_beta = pulay_beta

        assert self.__what in ["density", "potential"]
        assert self.__scheme in ["linear", "pulay", None]

        self.__val_stack = []
        self.__residual_stack = []
        return

    def __str__(self):
        s = str()
        s += "\n============================ [ Mixing ] ============================="
        s += "\n* what       : " + str(self.__what)
        s += "\n* scheme     : " + str(self.__scheme)
        s += "\n* parameter  : " + str(self.__parameter)
        if self.__scheme == "pulay":
            s += "\n* pulay_beta : " + str(self.__pulay_beta)
            s += "\n* history    : " + str(self.__history)
        s += "\n=====================================================================\n"
        return s

    def append(self, var):
        """Append input variable to self.__val_stack and make new mixed thing."""
        # append new mixed thing to __val_stack
        if len(self.__val_stack) >= 1:
            # R^{i} = n^{i}_{out} - n^{i}_{in} : residue of var(density or potential)
            self.__residual_stack.append(var - self.__val_stack[-1])
        mixed_var = self._get_mixed_var(var, self.__scheme)
        self.__val_stack.append(mixed_var)
        if len(self.__val_stack) > self.__history:
            self.__val_stack.pop(0)
            self.__residual_stack.pop(0)
        return

    def get_mixed_one(self):
        return self.__val_stack[-1]

    @timer
    def _get_mixed_var(self, var, scheme):
        if scheme == "linear":
            ret_var = self._calc_linear_mixing(var, alpha=self.__parameter)
        elif scheme == "pulay":
            assert self.__history >= 3
            if len(self.__residual_stack) < 3:
                ret_var = self._calc_linear_mixing(var, alpha=self.__parameter)
            else:
                ret_var = self._calc_pulay_mixing(
                    self.__pulay_beta, len(self.__residual_stack)
                )
        elif scheme is None:
            ret_var = self.__val_stack[-1]
        else:
            assert False, f"{scheme} is not available mixing scheme."
        return ret_var

    def _calc_linear_mixing(self, var, alpha=0.3):
        r"""
        \rho^{i+1}_{in} = \alpha \rho^{i}_{out} + (1 - \alpha) \rho^{i}_{in}
        """
        if len(self.__val_stack) >= 1:
            return alpha * var + (1 - alpha) * self.__val_stack[-1]
        else:
            return var

    def _calc_pulay_mixing(self, beta, history):
        ## Pulay(DIIS) Mixing ##
        nspins = self.__val_stack[0].shape[0]
        ret_var = torch.zeros_like(self.__val_stack[0])
        alpha = self._calc_pulay_coeffs(history)
        for i_s, i in product(range(nspins), range(history)):
            ret_var[i_s] += alpha[i_s][i] * (
                self.__val_stack[i][i_s] + beta * self.__residual_stack[i][i_s]
            )
        return ret_var

    def _calc_pulay_coeffs(self, history):
        nspins = self.__val_stack[0].shape[0]
        alpha = torch.zeros(nspins, history, dtype=torch.float64)

        ## Make residual matrix.
        residual_matrix = torch.ones(
            nspins, history + 1, history + 1, dtype=torch.float64
        )
        for i_s in range(nspins):
            for i, j in combinations_with_replacement(range(history), 2):
                residual_matrix[i_s][i, j] = (
                    self.__residual_stack[i][i_s] @ self.__residual_stack[j][i_s]
                )
                if i != j:
                    residual_matrix[i_s][j, i] = residual_matrix[i_s][i, j]
            residual_matrix[i_s][history, history] = 0

        ## Calculate alpha
        vec = torch.zeros(history + 1, dtype=torch.float64)
        vec[-1] = 1
        for i_s in range(nspins):
            alpha[i_s] = (torch.linalg.inv(residual_matrix[i_s]) @ vec)[:history]
        return alpha

    @property
    def what(self):
        return self.__what
