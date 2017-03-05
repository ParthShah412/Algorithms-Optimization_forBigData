import numpy as np
from MatFunc import MatFunc

a,Sn,Un,Vn = MatFunc()


A = np.random.randint(1,11,size=(15,1))
B = np.zeros((10,1),dtype=np.int)
B[9:10,:] = 1


m = np.dot(np.transpose(Un),A)
p = A-np.dot(Un,m)
Ra = np.zeros((1,1),dtype=np.float64)
Ra[:1,:1] = np.linalg.norm(p)
P = p/Ra

n = np.dot(Vn,B)
q = B - np.dot(np.transpose(Vn),n)
Rb = np.zeros((1,1),dtype=np.float64)
Rb[:1,:1] = np.linalg.norm(q)
Q = q/Rb


dummy1 = np.concatenate((m,Ra),axis=0)
dummy2 = np.concatenate((n,Rb),axis=0)
dummy3 = np.concatenate((Un,P),axis=1)
dummy4 = np.concatenate((np.transpose(Vn),Q),axis=1)


Sn1 = np.zeros((4,4),dtype=np.float64)
Sn1[:3,:3] = Sn
K = Sn1 + np.dot(dummy1,np.transpose(dummy2))

BigMat = np.dot(dummy3,np.dot(K,np.transpose(dummy4)))
print BigMat

New_a =  a + np.dot(A,np.transpose(B))

print np.linalg.norm(New_a - BigMat)
