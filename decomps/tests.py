import numpy as np
from decomps import *
from leastsq import *
import pytest
import itertools
from numpy.linalg import qr, svd, norm, matrix_rank, lstsq
from scipy.linalg import lu

def generate_tests_decomp():
    test_cases = []
    test_cases.append(np.random.rand(5, 5))
    test_cases.append(np.random.rand(20, 10))
    
    # test_cases.append(np.random.rand(200, 100))
    # test_cases.append(np.random.rand(2000, 1000))

    # rank_col_def = np.random.rand(100, 200)
    # rank_col_def[:,1] = rank_col_def[:,0] + rank_col_def[:,2]
    # test_cases.append(rank_col_def)

    # rank_row_def = np.random.rand(100, 200)
    # rank_row_def[:,1] = rank_row_def[:,0] + rank_row_def[:,2]
    # test_cases.append(rank_row_def)

    return test_cases

def generate_tests_leastsq():
    test_cases = []
    
    test_cases.append((np.random.rand(5, 5), np.random.rand(5,)))
    # test_cases.append((np.eye(5), np.random.rand(5,)))
    
    # for i in range(5,100,5):
    #     test_cases.append((np.random.rand(i, i), np.random.rand(i,)))

    return test_cases


def test_qr_decomps():
    np.set_printoptions(precision=4, suppress=True, edgeitems=200)
    test_cases = generate_tests_decomp()
    qr_cgs = lambda A : QR_by_GS(A, method='cgs')
    qr_mgs = lambda A : QR_by_GS(A, method='mgs')
    ql_cgs = lambda A : QL_by_GS(A, method='cgs')
    ql_mgs = lambda A : QL_by_GS(A, method='mgs')    

    qr_functions = [QR_by_Householder, QR_by_Givens, qr_cgs, qr_mgs]
    ql_functions = [QL_by_Givens, QL_by_Householder, ql_cgs, ql_mgs]

    for qr_func, matr in itertools.product(qr_functions, test_cases):
        Q, R, unitary_error = qr_func(matr)
        reconstruction_error = norm(matr - Q @ R) / norm(matr)
        triang_error = norm(R - np.triu(R)) / norm(matr)

        # print(np.diag(R))
        
        # assert triang_error < 10**-12
        assert unitary_error < 10**-12
        assert reconstruction_error < 10**-12

    for qr_func, matr in itertools.product(ql_functions, test_cases):
        Q, L, unitary_error = qr_func(matr)
        reconstruction_error = norm(matr - Q @ L) / norm(matr)
        triang_error = norm(L - np.tril(L)) / norm(matr)

        # print(np.diag(L))

        assert triang_error < 10**-12
        assert unitary_error < 10**-12
        assert reconstruction_error < 10**-12


def test_solve_ut():

    for i in range(2,100):
        R = lu(np.random.rand(i,i))[2]
        b = np.random.rand(i,)
        x = solve_ut(R,b)
        
        err = norm(R @ x - b)
        numpy_err = norm(R @ lstsq(R,b,rcond=None)[0] - b)
        err_diff = abs(err-numpy_err)*err
        assert err_diff < 10**-8

def test_solve_lt():

    for i in range(2,100):
        L = lu(np.random.rand(i,i))[1]
        b = np.random.rand(i,)
        x = solve_lt(L,b)
        
        err = norm(L @ x - b)
        numpy_err = norm(L @ lstsq(L,b,rcond=None)[0] - b)
        err_diff = abs(err-numpy_err)*err
        assert err_diff < 10**(-8)



def test_lu_decomps():
    np.set_printoptions(precision=4, suppress=True)
    test_cases = [case[0] for case in generate_tests_leastsq()]

    lu_partial = lambda A : lu_decomp(A, pivot='partial')
    lu_non = lambda A : lu_decomp(A, pivot='non')
    lu_partial_inv = lambda A : lu_decomp(A, pivot='partial', inverse=True)
    lu_non_inv = lambda A : lu_decomp(A, pivot='non', inverse=True)

    decomp_fns = [lu_non]#, lu_non_inv]
    decomp_pivot_fns = [lu_partial]#, lu_partial_inv]

    for decomp_fn, A in itertools.product(decomp_fns, test_cases):
        m, n = A.shape
        L, U = decomp_fn(A)

        reconstruction_error = norm(U @ L - A)

        assert reconstruction_error < 10**(-10)

    for decomp_fn, A in itertools.product(decomp_pivot_fns, test_cases):
        m, n = A.shape
        L, U, P = decomp_fn(A)

        reconstruction_error = norm(U @ L - P @ A)

        assert reconstruction_error < 10**(-10)        
        

def test_leastsq():
    np.set_printoptions(precision=4, suppress=True)
    test_cases = generate_tests_leastsq()

    lu_partial = lambda A,b : lstsq_by_lu(A, b, pivot='partial')
    lu_non = lambda A,b : lstsq_by_lu(A, b, pivot='non')
    lu_partial_inv = lambda A,b : lstsq_by_lu(A, b, pivot='partial', inverse=True)
    lu_non_inv = lambda A,b : lstsq_by_lu(A, b, pivot='non', inverse=True)

    lstsq_fns = [lstsq_by_QR, lu_partial, lu_non]#, lu_partial_inv, lu_non_inv]
    
    for lst, (A, b) in itertools.product(lstsq_fns, test_cases):
        x = lst(A, b)
        x_numpy = lstsq(A,b,rcond=None)[0]

        err = norm(A @ x - b)
        numpy_err = norm(A @ x_numpy - b)
        err_diff = abs(err - numpy_err)
        # print('least square error = {}'.format(err))
        # print('numpy error = {}'.format(numpy_err))
        # print('error diff = {}'.format(err_diff))
        
        assert err < 10**-8
