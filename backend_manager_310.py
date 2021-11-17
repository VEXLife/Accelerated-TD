#!python
# -*- coding: utf-8 -*-
# @Author：BWLL
# A helper script for backend switching, requiring Python 3.10 later

import os
import warnings
from typing import TypeAlias, Union

if "ATD_BACKEND" not in os.environ:
    os.environ["ATD_BACKEND"] = "NumPy"

match os.environ["ATD_BACKEND"]:
    case "NumPy" | "numpy" | "Numpy":
        try:
            import numpy as Backend

            Matrix: TypeAlias = Backend.ndarray
            Fraction: Union = int | float | Backend.floating | Backend.integer
            Backend.create_matrix_func = Backend.array
            Backend.convert_to_matrix_func = Backend.asarray

            if Backend.__version__ < "1.19.0":
                warnings.warn(f"NumPy {Backend.__version__} might not work. You'd better upgrade to a newer version.",
                              category=ImportWarning)
        except Exception:
            raise ImportError("Unable to import NumPy!")
        else:
            print("Successfully initialized NumPy.\nUsing NumPy as Backend.")
    case "PyTorch" | "pytorch" | "Torch" | "torch":
        try:
            import torch as Backend
            import torch.nn.functional as F
            from itertools import chain

            Matrix: TypeAlias = Backend.Tensor
            Fraction: Union = int | float | Backend.FloatType | Backend.IntType
            Backend.create_matrix_func = Backend.tensor
            Backend.convert_to_matrix_func = Backend.as_tensor
            Backend.pad = lambda mat, padding: F.pad(mat, pad=tuple(chain.from_iterable(padding)), mode="constant", value=0)

            if Backend.__version__ < "1.10":
                warnings.warn(f"PyTorch {Backend.__version__} might not work. You'd better upgrade to a newer version.",
                              category=ImportWarning)
        except Exception:
            raise ImportError("Unable to import PyTorch!")
        else:
            print("Successfully initialized PyTorch.\nUsing PyTorch as Backend.")
        # FIXME PyTorch doesn't support Python 3.10 yet
    case _:
        raise ImportError("The required backend is not supported!")
