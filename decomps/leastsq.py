import numpy as np
from numpy.linalg import qr, svd, norm, matrix_rank


def lstsq_by_QR(A, b):

    m, n = A.shape
    Q = np.eye(m, m)
    R = A[:m,:n].copy()
    bhat = b.copy()

    for i in range(n-1):
        l = R[i:, i]
        v = np.sign(l[0]) * norm(l) * np.eye(l.shape[0],1)[:,0].T + l
        v /= norm(v)
        # Q[:,i:] -= 2 * (Q[:,i:] @ v)[:,np.newaxis] @ v[np.newaxis,:]
        R[i:,i:n] -= 2 * v[:,np.newaxis] @ (v @ R[i:,i:n])[np.newaxis,:]
        bhat[i:] -= 2 * v * (v @ bhat[i:])

    x = solve_ut(R[:n,:n],bhat)
        
    return x


def lu_decomp(A, pivot='partial', inverse=False):

    m, n = A.shape

    if m != n:
        raise ValueError('LU decomposition only implemented for square matrices.')

    U = np.eye(n)
    L = A.copy()
    P = np.eye(n)

    if inverse:

        for i in range(n-1, 0, -1):

            if pivot == 'partial':
                pass
            
            # U[:i,i] -= 
                

    else:
        for i in range(n-1, 0, -1):

            if pivot == 'partial':
                ind = np.argmax(abs(L[:i+1,i]))
                
                if ind != i:
                
                    tmp = L[ind,:i+1].copy()
                    L[ind,:i+1] = L[i,:i+1]
                    L[i,:i+1] = tmp

                    tmp = U[ind,i+1:].copy()
                    U[ind,i+1:] = U[i,i+1:]
                    U[i,i+1:] = tmp

                    tmp = P[ind,:].copy()
                    P[ind,:] = P[i,:]
                    P[i,:] = tmp
        
            U[:i,i] = L[:i,i] / L[i,i]
            L[:i,:i+1] -= U[:i,i][:,np.newaxis] @ L[i,:i+1][np.newaxis,:]

    #         print('U = \n{} \nL = \n{}'.format(U,L))

    # print('U = \n{} \nU^-1 = \n{}'.format(U, np.linalg.inv(U)))
            
    if pivot == 'partial':
        return L, U, P
    else:
        return L, U


def lstsq_by_lu(A, b, pivot='partial', inverse=False):

    m, n = A.shape

    if m != n:
        raise ValueError('LU decomposition only implemented for square matrices.')

    U = np.eye(n)
    L = A.copy()
    bhat = b.copy()
    
    for i in range(n-1, 0, -1):

        if pivot == 'partial':
            ind = np.argmax(abs(L[:i+1,i]))
                
            if ind != i:
                
                tmp = L[ind,:i+1].copy()
                L[ind,:i+1] = L[i,:i+1]
                L[i,:i+1] = tmp
                
                tmp = U[ind,i+1:].copy()
                U[ind,i+1:] = U[i,i+1:]
                U[i,i+1:] = tmp
                
                tmp = bhat[ind]
                bhat[ind] = bhat[i]
                bhat[i] = tmp
        
        U[:i,i] = L[:i,i] / L[i,i]
        L[:i,:i+1] -= U[:i,i][:,np.newaxis] @ L[i,:i+1][np.newaxis,:]
        bhat[:i] -= U[:i,i] * bhat[i]

    x = solve_lt(L,bhat)
    
    return x


def solve_ut(R,b):

    n = R.shape[1]
    x = np.zeros((n,))
    
    x[n-1] = b[n-1] / R[n-1,n-1]
    for i in range(n-2,-1, -1):
        x[i] = (b[i] - x[i+1:n] @ R[i,i+1:n]) / R[i,i]
    
    return x


def solve_lt(L,b):
    
    n = L.shape[1]
    x = np.zeros((n,))
    
    x[0] = b[0] / L[0,0]
    for i in range(1,n):
        x[i] = (b[i] - x[:i] @ L[i,:i]) / L[i,i]
    
    return x
