import numpy as np
from numpy.linalg import qr, svd, norm, matrix_rank


def QR_by_GS(A, method):

    m, n = A.shape
    Q = np.zeros((m,n)) if method =='cgs' else A.copy()
    R = np.zeros((n,n))

    # R[0,0] = norm(A[:,0])
    # q = A[:,0] / R[0,0]
    # Q[:,0] = q

    for i in range(n):
        if method == 'cgs':
            R[:i,i] = Q[:,:i].T @ A[:,i]
            q = A[:,i] - Q[:,:i] @ R[:i,i]
            R[i,i] = norm(q)
            if R[i,i] == 0:
                raise ValueError('Matrix is rank deficient.')
            Q[:,i] = q / R[i,i]
            
        elif method == 'mgs':
            R[i,i] = norm(Q[:,i])
            Q[:,i] /= R[i,i]

            for j in range(i+1,n):
                R[i,j] = Q[:,j] @ Q[:,i]
                Q[:,j] = Q[:,j] - R[i,j] * Q[:,i]


            # R[i,i+1:] * Q[:,i]
            # R[i,i+1:] = Q[:,i].T @ Q[:,i+1:]
            # Q[:,i+1:] = Q[:,i+1:] - R[i,:] @ Q[:,i]
            
    return Q, R, (norm(np.eye(n) - Q.T @ Q)/norm(A), norm(A - Q @ R) / norm(A))


def QR_by_Householder(A):
    m, n = A.shape
    Q = np.eye(m, m)
    R = A[:m,:n].copy()

    for i in range(n-1):
        x = R[i:, i]
        # v = np.sign(x[0]) * norm(x) * np.eye(x.shape[0],1)[:,0].T + x
        v = - norm(x) * np.eye(x.shape[0],1)[:,0].T + x
        v /= norm(v)
        Q[:,i:] -= 2 * (Q[:,i:] @ v)[:,np.newaxis] @ v[np.newaxis,:]
        R[i:,i:n] -= 2 * v[:,np.newaxis] @ (v @ R[i:,i:n])[np.newaxis,:]

    # if R[n-1,n-1] < 0:
    #     R[n-1,n-1] = abs(R[n-1,n-1])
    
    return Q, R, (norm(np.eye(m) - Q.T @ Q)/norm(A), norm(A - Q @ R) / norm(A))


def QR_by_Givens(A):
    m, n = A.shape
    Q = np.eye(m,m)
    R = A.copy()
    
    for i in range(n):
        for j in range(i+1,m):
            Q_temp = np.eye(m,m)
            xi, xj = (R[i,i], R[j,i])
            if xj != 0:
                x_norm = norm(np.array([xi,xj]))
                Q_temp[i,i] = xi / x_norm
                Q_temp[j,i] = -xj / x_norm
                Q_temp[j,j] = xi / x_norm
                Q_temp[i,j] = xj / x_norm
                Q[:,:] = Q @ Q_temp.T
                R = Q_temp @ R
                
                # Q_block = Q[i:j+1:j-i, i:j+1:j-i]
                # R_block = R[i:j+1:j-i, i:j+1:j-i]
                # Q_temp = np.array([[xi,xj],[-xj,xi]])/x_norm
                # Q_block = Q_temp.T @ Q_block
                # R_block = Q_temp @ R_block
                # Q[i:j+1:j-i, i:j+1:j-i] = Q_block
                # R[i:j+1:j-i, i:j+1:j-i] = R_block
    
    return Q, R, (norm(np.eye(m) - Q.T @ Q)/norm(A), norm(A - Q @ R) / norm(A))


def QL_by_GS(A, method):

    m, n = A.shape
    Q = np.zeros((m,n)) if method =='cgs' else A.copy()
    L = np.zeros((n,n))

    for i in range(n-1, -1, -1):
        if method == 'cgs':
            L[i+1:,i] = Q[:,i+1:].T @ A[:,i]
            q = A[:,i] - Q[:,i+1:] @ L[i+1:,i]
            L[i,i] = norm(q)
            if L[i,i] == 0:
                raise ValueError('Matrix is rank deficient.')
            Q[:,i] = q / L[i,i]

        elif method == 'mgs':
            L[i,i] = norm(Q[:,i])
            Q[:,i] /= L[i,i]

            for j in range(i):
                L[i,j] = Q[:,j] @ Q[:,i]
                Q[:,j] = Q[:,j] - L[i,j] * Q[:,i]

    return Q, L, (norm(np.eye(n) - Q.T @ Q)/norm(A), norm(A - Q @ L) / norm(A))


def QL_by_Householder(A):
    m, n = A.shape
    Q = np.eye(m, m)
    L = A.copy()
    
    for i in range(n-1, 0, -1):
        Q_temp = np.eye(m,n)
        x = L[:i+1, i]
        v = - norm(x) * np.eye(x.shape[0],i+1)[:,i].T + x
        v /= norm(v)
        Q[:,:i+1] -= 2 * (Q[:,:i+1] @ v)[:,np.newaxis] @ v[np.newaxis,:]
        L[:i+1,:i+1] -= 2 * v[:,np.newaxis] @ (v @ L[:i+1,:i+1])[np.newaxis,:]

    return Q, L, (norm(np.eye(m) - Q.T @ Q), norm(A - Q @ L) / norm(A))


def QL_by_Givens(A):

    m, n = A.shape
    Q = np.eye(m,m)
    L = A.copy()
    
    for i in range(n-1,-1,-1):
        for j in range(i-1,-1,-1):
            Q_temp = np.eye(m,m)
            xi, xj = (L[i,i], L[j,i])
            if xj != 0:
                x_norm = norm(np.array([xi,xj]))
                Q_temp[i,i] = xi / x_norm
                Q_temp[j,i] = -xj / x_norm
                Q_temp[j,j] = xi / x_norm
                Q_temp[i,j] = xj / x_norm
                Q[:,:] = Q @ Q_temp.T
                L = Q_temp @ L
    
    return Q, L, (norm(np.eye(m,m) - Q.T @ Q)/norm(A), norm(A - Q @ L) / norm(A))
