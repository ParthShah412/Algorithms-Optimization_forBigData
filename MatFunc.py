import numpy as np


def MatFunc():
    B = np.arange(1,151,1)
    a = B.reshape((15,10))

    U, s, V = np.linalg.svd(a,full_matrices=True)

    S = np.zeros((15,10),dtype=np.float64)
    S[:10,:10] = np.diag(s)


    Sn = S[:3,:3]
    Un = U[:,:3]
    Vn = V[:3,:]

    return a,Sn,Un,Vn
