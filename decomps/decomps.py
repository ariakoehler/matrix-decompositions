import numpy as np
from numpy.linalg import qr, svd, norm, matrix_rank


def QR_by_GS(A, method):

    m, n = A.shape
    Q = np.zeros((m,n)) if method =='cgs' else A.copy()
    R = np.zeros((n,n))

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
            
    return Q, R, (norm(np.eye(n) - Q.T @ Q)/norm(A), norm(A - Q @ R) / norm(A))


def QR_by_Householder(A):
    m, n = A.shape
    Q = np.eye(m, m)
    R = A[:m,:n].copy()

    for i in range(n-1):
        x = R[i:, i]
        v = - norm(x) * np.eye(x.shape[0],1)[:,0].T + x
        v /= norm(v)
        Q[:,i:] -= 2 * (Q[:,i:] @ v)[:,np.newaxis] @ v[np.newaxis,:]
        R[i:,i:n] -= 2 * v[:,np.newaxis] @ (v @ R[i:,i:n])[np.newaxis,:]
    
    return Q, R, (norm(np.eye(m) - Q.T @ Q)/norm(A), norm(A - Q @ R) / norm(A))


def QR_by_Givens(A):
    m, n = A.shape
    Q = np.eye(m,m)
    R = A.copy()
    
    for i in range(n):
        for j in range(i+1,m):
            Q_temp = np.eye(2)
            xi, xj = (R[i,i], R[j,i])
            if xj != 0:
                x_norm = norm(np.array([xi,xj]))
                Q_temp[0,0] = xi / x_norm
                Q_temp[1,0] = -xj / x_norm
                Q_temp[1,1] = xi / x_norm
                Q_temp[0,1] = xj / x_norm
                Q[:,i:j+1:j-i] = Q[:,i:j+1:j-i] @ Q_temp.T
                R[i:j+1:j-i,:] = Q_temp @ R[i:j+1:j-i,:]
    
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
            Q_temp = np.eye(2)
            xi, xj = (L[i,i], L[j,i])
            if xj != 0:
                x_norm = norm(np.array([xi,xj]))
                Q_temp[0,0] = xi / x_norm
                Q_temp[1,0] = -xj / x_norm
                Q_temp[1,1] = xi / x_norm
                Q_temp[0,1] = xj / x_norm
                Q[:,j:i+1:abs(j-i)] = Q[:,j:i+1:abs(j-i)] @ Q_temp.T
                L[j:i+1:abs(j-i),:] = Q_temp @ L[j:i+1:abs(j-i),:]
                
    return Q[:m,:n], L[:n,:n], (norm(np.eye(m,m) - Q.T @ Q)/norm(A), norm(A - Q @ L) / norm(A))
