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
# A helper script for backend switching.

import os
import warnings
from typing import TypeAlias, Union

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

        Matrix: TypeAlias = Backend.ndarray
        Decimal: TypeAlias = Union[int, float, Backend.floating, Backend.integer]
        Backend.create_matrix_func = Backend.array
        Backend.convert_to_matrix_func = Backend.asarray


        def extend_with_000(mat: Matrix) -> Matrix:
            """
            Pad 0 around the matrix.
            """
            return Backend.pad(mat, ((0, 1), (0, 1)))


        def extend_with_010(mat: Matrix) -> Matrix:
            """
            Setting the bottom right number to 1.
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

        Matrix: TypeAlias = Backend.Tensor
        Decimal: TypeAlias = Union[int, float, Backend.FloatType, Backend.IntType]
        Backend.create_matrix_func = Backend.tensor
        Backend.convert_to_matrix_func = Backend.as_tensor


        def extend_with_000(mat: Matrix) -> Matrix:
            """
            Pad 0 around the matrix.
            """
            return F.pad(mat, pad=(0, 1, 0, 1), mode="constant", value=0)


        def extend_with_010(mat: Matrix) -> Matrix:
            """
            Setting the bottom right number to 1.
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
