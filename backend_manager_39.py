#!python
# -*- coding: utf-8 -*-
# @Author：Midden Vexu
# A helper script for backend switching.

import os
import warnings
from typing import Union

if "ATD_BACKEND" not in os.environ:
    os.environ["ATD_BACKEND"] = "NumPy"


# makes Python 3.9 or earlier support checking if an object is belong to a typing.Union directly.
def _isinstance(obj, class_or_tuple):
    try:
        return original_isinstance(obj, class_or_tuple)
    except TypeError:
        return original_isinstance(obj, class_or_tuple.__args__)


original_isinstance = isinstance
isinstance = _isinstance

if os.environ["ATD_BACKEND"] == "NumPy":
    try:
        import numpy as Backend

        Matrix = Backend.ndarray
        Fraction: Union = Union[int, float, Backend.floating, Backend.integer]
        Backend.create_matrix_func = Backend.array
        Backend.convert_to_matrix_func = Backend.asarray


        def extend_with_000(mat: Matrix) -> Matrix:
            """
            用于在二维张量mat周围补0
            """
            return Backend.pad(mat, ((0, 1), (0, 1)))


        def extend_with_010(mat: Matrix) -> Matrix:
            """
            在补0的基础上将右下角设为1
            """
            mat_ = extend_with_000(mat)
            mat_[-1, -1] = 1
            return mat_


        if Backend.__version__ < "1.19.0":
            warnings.warn(f"NumPy {Backend.__version__} might not work. You'd better upgrade to a newer version.",
                          category=ImportWarning)
    except Exception:
        raise ImportError("Unable to import NumPy!")
    else:
        print("Successfully initialized NumPy.\nUsing NumPy as Backend.")
elif os.environ["ATD_BACKEND"] == "PyTorch":
    try:
        import torch as Backend
        import torch.nn.functional as F
        from itertools import chain

        Matrix = Backend.Tensor
        Fraction: Union = Union[int, float, Backend.FloatType, Backend.IntType]
        Backend.create_matrix_func = Backend.tensor
        Backend.convert_to_matrix_func = Backend.as_tensor


        def extend_with_000(mat: Matrix) -> Matrix:
            """
            用于在二维张量mat周围补0
            """
            return F.pad(mat, pad=(0, 1, 0, 1), mode="constant", value=0)


        def extend_with_010(mat: Matrix) -> Matrix:
            """
            在补0的基础上将右下角设为1
            """
            mat_ = extend_with_000(mat)
            mat_[-1, -1] = 1
            return mat_


        if Backend.__version__ < "1.10":
            warnings.warn(f"PyTorch {Backend.__version__} might not work. You'd better upgrade to a newer version.",
                          category=ImportWarning)
    except Exception:
        raise ImportError("Unable to import PyTorch!")
    else:
        print("Successfully initialized PyTorch.\nUsing PyTorch as Backend.")
else:
    raise ImportError("The required backend is not supported!")
