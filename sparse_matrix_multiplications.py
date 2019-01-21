import numpy as np
import scipy as sp
import scipy.sparse
from ctypes import POINTER, c_int, c_char, c_double, byref, cdll

mkl = cdll.LoadLibrary("libmkl_rt.dylib")


def mkl_csr_matvec(A, x):
    """
    Parameters
    ----------
    A : scipy.sparse csr matrix
    x : numpy 1d array
    """

    if not sp.sparse.isspmatrix_csr(A):
        raise TypeError("The matrix must be a scipy sparse CSR matrix.")

    if x.ndim != 1:
        raise TypeError("The vector to be multiplied must be a 1d array.")

    if x.dtype.type is not np.double:
        x = x.astype(np.double, copy=True)

    # Allocate the result of the matrix-vector multiplication.
    result = np.empty(A.shape[0])

    # Get pointers to the numpy arrays.
    data_ptr = A.data.ctypes.data_as(POINTER(c_double))
    indptr_ptr = A.indptr.ctypes.data_as(POINTER(c_int))
    indices_ptr = A.indices.ctypes.data_as(POINTER(c_int))
    x_ptr = x.ctypes.data_as(POINTER(c_double))
    result_ptr = result.ctypes.data_as(POINTER(c_double))

    transpose_flag = byref(c_char(bytes('n', 'utf-8')))
    result_length = byref(c_int(result.size))
    mkl.mkl_cspblas_dcsrgemv(
        transpose_flag, result_length,
        data_ptr, indptr_ptr, indices_ptr, x_ptr, result_ptr
    )

    return result