import numpy as np
from scipy.linalg import hilbert
from decomps import *
from leastsq import *
import matplotlib.pyplot as plt


def generate_error(decomp_fns, decomp_names, m_range):
    
    unit_errors = dict([(decomp, []) for decomp in decomp_names])

    factor_errors = dict([(decomp, []) for decomp in decomp_names])

    n = 100
    for m in m_range:
        
        A = np.random.rand(m, n)

        for decomp, name in zip(decomp_fns, decomp_names):
            print(name, m)
            unit, R, (unit_error, factor_error) = decomp(A)
            unit_errors[name].append(unit_error)
            factor_errors[name].append(factor_error)

    return unit_errors, factor_errors

if __name__=='__main__':
    qr_cgs = lambda A : QR_by_GS(A, method='cgs')
    qr_mgs = lambda A : QR_by_GS(A, method='mgs')
    ql_cgs = lambda A : QL_by_GS(A, method='cgs')
    ql_mgs = lambda A : QL_by_GS(A, method='mgs')
    
    decomp_fns = [QR_by_Householder, QR_by_Givens, qr_cgs, qr_mgs, QL_by_Givens, QL_by_Householder, ql_cgs, ql_mgs]

    decomp_names = ['QR_by_Householder', 'QR_by_Givens', 'qr_cgs', 'qr_mgs', 'QL_by_Givens', 'QL_by_Householder', 'ql_cgs', 'ql_mgs']

    m_range = range(200,1000,100)

    unit_errors, factor_errors = generate_error(decomp_fns, decomp_names, m_range)

    plt.figure()

    for decomp in decomp_names:
        plt.plot(m_range, unit_errors[decomp], label=decomp)
        
    plt.legend(loc='best')
    plt.title('Orthogonality Error')
    plt.show()


    plt.figure()

    for decomp in decomp_names:
        plt.plot(m_range, factor_errors[decomp], label=decomp)
        
    plt.legend(loc='best')
    plt.title('Factorization Error')
    plt.show()
