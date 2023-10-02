import math
import numpy as np

import cupy as cp
from jinja2 import Template
from symengine import sympify

from hastypy.opt.sym import util

from hastypy.opt.cuda import linalg
from hastypy.opt.cuda import solver

from hastypy.opt.cuda import cuda_program as cudap
from hastypy.opt.cuda.cuda_program import CudaFunction, CudaTensor
from hastypy.opt.cuda.cuda_program import CudaTensorChecking as ctc


def bpolyfit(x: cp.ndarray, y: cp.ndarray, deg: int, dtype: cp.dtype):
    ndata = y.shape[0]
    nprobs = y.shape[1]
    A = cp.ones((nprobs, ndata, deg+1), dtype=dtype)
    for i in range(1,deg+1):
        A[:,:,i] = (x ** i).squeeze(0).transpose()
    
    pA = cp.linalg.pinv(A)
    return (pA @ cp.expand_dims(y.transpose(), axis=2)).squeeze(-1).transpose()

