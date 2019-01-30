import numpy as np
cimport numpy as np
from cython cimport view


cdef extern from "mkl.h":

	ctypedef int MKL_INT

	ctypedef enum sparse_index_base_t:
		SPARSE_INDEX_BASE_ZERO = 0
		SPARSE_INDEX_BASE_ONE = 1

	ctypedef enum sparse_status_t:
		SPARSE_STATUS_SUCCESS = 0 # the operation was successful
		SPARSE_STATUS_NOT_INITIALIZED = 1 # empty handle or matrix arrays
		SPARSE_STATUS_ALLOC_FAILED = 2 # internal error: memory allocation failed
		SPARSE_STATUS_INVALID_VALUE = 3 # invalid input value
		SPARSE_STATUS_EXECUTION_FAILED = 4 # e.g. 0-diagonal element for triangular solver, etc.
		SPARSE_STATUS_INTERNAL_ERROR = 5 # internal error
		SPARSE_STATUS_NOT_SUPPORTED = 6 # e.g. operation for double precision doesn't support other types */

	ctypedef enum sparse_operation_t:
		SPARSE_OPERATION_NON_TRANSPOSE = 10
		SPARSE_OPERATION_TRANSPOSE = 11
		SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12

	ctypedef enum sparse_matrix_type_t:
		SPARSE_MATRIX_TYPE_GENERAL = 20 # General case
		SPARSE_MATRIX_TYPE_SYMMETRIC = 21 # Triangular part of the matrix is to be processed
		SPARSE_MATRIX_TYPE_HERMITIAN = 22
		SPARSE_MATRIX_TYPE_TRIANGULAR = 23
		SPARSE_MATRIX_TYPE_DIAGONAL = 24 # diagonal matrix; only diagonal elements will be processed
		SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR = 25
		SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL = 26 # block-diagonal matrix; only diagonal blocks will be processed

	ctypedef enum sparse_fill_mode_t:
		SPARSE_FILL_MODE_LOWER = 40 # lower triangular part of the matrix is stored
		SPARSE_FILL_MODE_UPPER = 41 # upper triangular part of the matrix is stored
		SPARSE_FILL_MODE_FULL = 42 # upper triangular part of the matrix is stored

	ctypedef enum sparse_diag_type_t:
		SPARSE_DIAG_NON_UNIT = 50 # triangular matrix with non-unit diagonal
		SPARSE_DIAG_UNIT = 51 # triangular matrix with unit diagonal

	struct sparse_matrix:
		pass

	ctypedef sparse_matrix* sparse_matrix_t

	struct matrix_descr:
		sparse_matrix_type_t type # matrix type: general, diagonal or triangular / symmetric / hermitian
		sparse_fill_mode_t mode # upper or lower triangular part of the matrix ( for triangular / symmetric / hermitian case)
		sparse_diag_type_t diag # unit or non-unit diagonal ( for triangular / symmetric / hermitian case)

	sparse_status_t mkl_sparse_d_create_csr(
		sparse_matrix_t* A,
		const sparse_index_base_t indexing, # indexing: C-style or Fortran-style
		const MKL_INT rows,
		const MKL_INT cols,
		MKL_INT *rows_start,
		MKL_INT *rows_end,
		MKL_INT *col_indx,
		double *values
	)

	sparse_status_t mkl_sparse_d_mv(
		sparse_operation_t operation,
		double alpha,
		const sparse_matrix_t A,
		matrix_descr descr,
		const double *x,
		double beta,
		double *y
	)

	sparse_status_t mkl_sparse_spmm(
		sparse_operation_t operation,
		const sparse_matrix_t A,
		const sparse_matrix_t B,
		sparse_matrix_t *C
	)

cdef struct matrix_descr:
	sparse_matrix_type_t type


def mkl_csr_matvec(A_csr, x, transpose=False):
	A = to_mkl_csr(A_csr)
	result = np.zeros(A_csr.shape[transpose])
	cdef double[:] x_view = x
	cdef double[:] result_view = result
	matvec_status = mkl_csr_plain_matvec(
		A, &x_view[0], &result_view[0], int(transpose)
	)
	return result


def mkl_csr_matmat(A_csr, B_csr):
	# cdef bint transpose_flag = int(transpose)
	cdef sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE
	cdef sparse_matrix_t C
	cdef sparse_matrix_t A = to_mkl_csr(A_csr)
	cdef sparse_matrix_t B = to_mkl_csr(B_csr)
	mkl_sparse_spmm(operation, A, B, &C)

# TODO: create a class to hold the pointer to the MKL sparse matrix?


cdef sparse_matrix_t to_mkl_csr(A_csr):

	cdef MKL_INT rows = A_csr.shape[0]
	cdef MKL_INT cols = A_csr.shape[1]
	cdef sparse_matrix_t A
	cdef sparse_index_base_t base_index=SPARSE_INDEX_BASE_ZERO

	cdef MKL_INT[:] row_ptr_array = A_csr.indptr
	cdef MKL_INT[:] col_index_array = A_csr.indices
	cdef double[:] value_array = A_csr.data

	cdef MKL_INT* rows_start = &row_ptr_array[0]
	cdef MKL_INT* rows_end = &row_ptr_array[1]
	cdef MKL_INT* col_index = &col_index_array[0]
	cdef double* values = &value_array[0]

	create_status = mkl_sparse_d_create_csr(
		&A, base_index, rows, cols,
		rows_start, rows_end, col_index, values
	)
	return A


cdef mkl_csr_plain_matvec(
		sparse_matrix_t A, const double* x, double* result, bint transpose
	):

	cdef sparse_operation_t operation
	if transpose:
		operation = SPARSE_OPERATION_TRANSPOSE
	else:
		operation = SPARSE_OPERATION_NON_TRANSPOSE
	cdef double alpha = 1.
	cdef double beta = 0.
	cdef matrix_descr mat_descript
	mat_descript.type = SPARSE_MATRIX_TYPE_GENERAL
	status = mkl_sparse_d_mv(
		operation, alpha, A, mat_descript, x, beta, result
	)
	return status
