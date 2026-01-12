"""
/home/jhwoo/.conda/envs/gospel/lib/python3.7/site-packages/scipy-1.6.1-py3.7-linux-x86_64.egg/scipy/sparse/linalg/interface.py
"""
from typing import Callable, Union, Tuple, Optional, List
import numpy as np
from scipy.sparse import isspmatrix
import torch


def aslinearoperator(A):
    """Return A as a LinearOperator.

    :type  A: np.ndarray, scipy.sparse, LinearOperator
    :param A:

    :rtype: LinearOperator
    :return:
        LinearOperator class object
    """
    if A is None:
        return None
    elif isinstance(A, LinearOperator) or isinstance(A, MultiTypeLinearOperator):
        return A
    elif isinstance(A, np.ndarray) or isinstance(A, torch.Tensor) or isspmatrix(A):
        if A.ndim != 2:
            raise ValueError("array must have ndim == 2")
        matvec = lambda x: A @ x
        return LinearOperator(A.shape, matvec, A.dtype)
    else:
        raise ValueError(f"{type(A)} is not supported type.")


class LinearOperator:
    """user-specified operations.

    :type  shape: torch.Tensor
    :param shape:
        shape of operator
    :type  matvec: function
    :param matvec:
        user-specified 'matrix-vector multiplication' operation
    :type  dtype: type
    :param dtype:
        Data type of the matrix

    **Example**

        >>> from gospel.LinearOperator import LinearOperator
        >>> # Define a simple matrix-vector multiplication function
        >>> f = lambda x: x * 2  # Replace with your operation
        >>> H = LinearOperator(shape=(100, 100), matvec=f, dtype=torch.float32, name="MyOperator")
    """

    def __init__(
        self,
        shape: Union[Tuple[int, ...], torch.Tensor],
        matvec: Callable[[torch.Tensor], torch.Tensor],
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
    ) -> None:
        self.shape = shape
        self.matvec = matvec
        self.dtype = dtype
        self.name = name
        # self.traced_function = self.matvec
        return

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.dot(x)

    def dot(self, x: torch.Tensor) -> torch.Tensor:
        return self.matvec(x)

    def __mul__(self, x):
        return self.dot(x)

    def __matmul__(self, x):
        return self.dot(x)

    # def trace(self, example_inputs):
    #     self.traced_function = torch.jit.trace(self.matvec, example_inputs)
    #     return


class MultiTypeLinearOperator:
    """
    A composite linear operator that selects an appropriate LinearOperator
    based on the data type of the input tensor.

    Parameters:
        operator_list (List[LinearOperator]):
            A list of LinearOperator objects.
        name (Optional[str]):
            An optional name for this composite operator.
    """

    def __init__(
        self, operator_list: List[LinearOperator], name: Optional[str] = None
    ) -> None:
        # Ensure all operators have the same shape.
        shape_list = [tuple(op.shape) for op in operator_list]
        if not all(shape == shape_list[0] for shape in shape_list):
            raise ValueError(
                f"All operators must have the same shape, but got {shape_list}"
            )

        self.operator_list = operator_list
        self.name = name
        self.dtype_list = [op.dtype for op in operator_list]
        self.name_list = [op.name for op in operator_list]
        return

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.dot(x)

    def dot(self, x: torch.Tensor, op_name: Optional[str] = None) -> torch.Tensor:
        """
        Perform matrix-vector multiplication using the appropriate LinearOperator.

        Parameters:
            x (torch.Tensor): The input tensor.
            op_name (Optional[str]): The operator name to select a specific operator, if needed.

        Returns:
            torch.Tensor: The result of the matrix-vector multiplication.
        """
        op = self.get_operator(x.dtype, op_name)
        return op.dot(x)

    def get_operator(
        self, dtype: torch.dtype, op_name: Optional[str] = None
    ) -> LinearOperator:
        """
        Retrieve a LinearOperator based on the specified dtype and optional operator name.

        Parameters:
            dtype (torch.dtype): The data type of the desired operator.
            op_name (Optional[str]): The name of the operator to retrieve.
                                     If not provided, the method expects that only one
                                     operator exists for the given dtype.

        Returns:
            LinearOperator: The operator that matches the given dtype (and op_name if provided).

        Raises:
            ValueError: If no matching operator is found, or if multiple operators exist for
                        the given dtype when op_name is not specified.
        """
        if op_name is not None:
            # Search for an operator matching both dtype and op_name.
            for op in self.operator_list:
                if op.dtype == dtype and op.name == op_name:
                    return op
            raise ValueError(
                f"No operator found with dtype {dtype} and name '{op_name}'."
            )
        else:
            # If op_name is not provided, select all operators matching the dtype.
            matching_ops = [op for op in self.operator_list if op.dtype == dtype]
            if not matching_ops:
                raise ValueError(f"No operator found with dtype {dtype}.")
            if len(matching_ops) > 1:
                raise ValueError(
                    f"Multiple operators found with dtype {dtype}. Please specify an operator name. "
                    f"Found operator names: {[op.name for op in matching_ops]}"
                )
            return matching_ops[0]

    def __mul__(self, x):
        return self.dot(x)

    def __matmul__(self, x):
        return self.dot(x)

    def __repr__(self) -> str:
        """
        Return a string representation of the MultiTypeLinearOperator.

        This representation includes the operator's name, shape (taken from the first operator),
        and a summary of each underlying operator's data type and name.
        """
        # Summarize each operator's data type and name.
        operator_info = ", ".join(
            f"(dtype={op.dtype}, name={op.name})" for op in self.operator_list
        )
        # Use the shape of the first operator as the representative shape.
        shape_repr = tuple(self.operator_list[0].shape) if self.operator_list else None
        return f"MultiTypeLinearOperator(name={self.name}, shape={shape_repr}, operators=[{operator_info}])"


if __name__ == "__main__":
    # Example matvec function definitions
    def matvec_float32(x: torch.Tensor) -> torch.Tensor:
        # Perform operations in float32
        return x * 2.0  # Example

    def matvec_float64(x: torch.Tensor) -> torch.Tensor:
        # Perform operations in float64
        return x + 10.0  # Example

    # Create LinearOperator instances for each dtype
    op32 = LinearOperator(
        shape=(100, 100),
        matvec=matvec_float32,
        dtype=torch.float32,
        name="operator_float32",
    )
    op64 = LinearOperator(
        shape=(100, 100),
        matvec=matvec_float64,
        dtype=torch.float64,
        name="operator_float64",
    )

    # Manage multiple LinearOperators using MultiTypeLinearOperator
    multi_op = MultiTypeLinearOperator(
        operator_list=[op32, op64],
        name="multi_operator_example",
    )
    print(multi_op)

    # Dynamically select the appropriate operator based on input tensor dtype
    x32 = torch.randn((100,), dtype=torch.float32)
    x64 = torch.randn((100,), dtype=torch.float64)

    y32 = multi_op.dot(x32)  # Internally uses op32
    y64 = multi_op.dot(x64)  # Internally uses op64

    print(y32.dtype, y64.dtype)  # Expected output: float32, float64
