import numpy as np
from scipy.linalg import hilbert
from decomps import *
from leastsq import *
import matplotlib.pyplot as plt

import os


def generate_qr(decomp_fns, decomp_names, m_range):

    Qs = dict([(decomp, []) for decomp in decomp_names])
    RLs = dict([(decomp, []) for decomp in decomp_names])
    unit_errors = dict([(decomp, []) for decomp in decomp_names])
    factor_errors = dict([(decomp, []) for decomp in decomp_names])

    n = 100
    for m in m_range:
        
        A = np.random.rand(m, n)

        for decomp, name in zip(decomp_fns, decomp_names):
            Q, RL, (unit_error, factor_error) = decomp(A)
            Qs[name].append(Q)
            RLs[name].append(RL)
            unit_errors[name].append(unit_error)
            factor_errors[name].append(factor_error)

    return unit_errors, factor_errors


def plot_qr_errors():
    qr_cgs = lambda A : QR_by_GS(A, method='cgs')
    qr_mgs = lambda A : QR_by_GS(A, method='mgs')
    ql_cgs = lambda A : QL_by_GS(A, method='cgs')
    ql_mgs = lambda A : QL_by_GS(A, method='mgs')
    
    qr_fns = [QR_by_Givens, QR_by_Householder, qr_cgs, qr_mgs]
    ql_fns = [QL_by_Givens, QL_by_Householder, ql_cgs, ql_mgs]

    qr_names = ['QR_by_Givens', 'QR_by_Householder', 'qr_cgs', 'qr_mgs']
    ql_names = ['QL_by_Givens', 'QL_by_Householder', 'ql_cgs', 'ql_mgs']

    m_range = range(200,1000,100)

    Qrs, qRs, qr_unit_errors, qr_factor_errors = generate_qr(qr_fns, qr_names, m_range)
    Qls, qLs, ql_unit_errors, ql_factor_errors = generate_qr(ql_fns, ql_names, m_range)
    
    plt.figure()

    for decomp in qr_names:
        plt.plot(m_range, qr_unit_errors[decomp], label=decomp)
        
    plt.legend(loc='best')
    plt.title('QR Orthogonality Error')
    plt.savefig(os.path.join('figs','qr-orthog.png'))


    plt.figure()

    for decomp in qr_names:
        plt.plot(m_range, qr_factor_errors[decomp], label=decomp)
        
    plt.legend(loc='best')
    plt.title('QR Factorization Error')
    plt.savefig(os.path.join('figs','qr-factor.png'))


    for decomp in ql_names:
        plt.plot(m_range, ql_unit_errors[decomp], label=decomp)
        
    plt.legend(loc='best')
    plt.title('QL Orthogonality Error')
    plt.savefig(os.path.join('figs','ql-orthog.png'))


    plt.figure()

    for decomp in ql_names:
        plt.plot(m_range, ql_factor_errors[decomp], label=decomp)
        
    plt.legend(loc='best')
    plt.title('QL Factorization Error')
    plt.savefig(os.path.join('figs','ql-factor.png'))


    for m in m_range:
        


def plot_qr_hilb():
    pass

    
if __name__=='__main__':
    plot_qr_errors()
    plot_qr_hilb()
