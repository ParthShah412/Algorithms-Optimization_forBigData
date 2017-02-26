import numpy as np

a = np.random.randint(1,11,size=(15,10))
print a

U, s, V = np.linalg.svd(a,full_matrices=True)

S = np.zeros((15,10),dtype=np.float64)
S[:10,:10] = np.diag(s)


Sn = S[:3,:3]
Un = U[:,:3]
Vn = V[:3,:]

A = np.random.randint(1,11,size=(15,4))
B = np.random.randint(1,11,size=(10,4))

P,Ra = np.linalg.qr(np.dot((np.eye(15)-np.dot(Un,np.transpose(Un))),A))
Q,Rb = np.linalg.qr(np.dot((np.eye(10)-np.dot(np.transpose(Vn),Vn)),B))

Sn1 = np.zeros((7,7),dtype=np.float64)
Sn1[:3,:3] = Sn

dummy1 = np.zeros((7,4),dtype=np.float64)
dummy1[:3,:] = np.dot(np.transpose(Un),A)
dummy1[3:,:] = Ra
print dummy1.shape

dummy2 = np.zeros((7,4),dtype=np.float64)
dummy2[:3,:] = np.dot(Vn,B)
dummy2[3:,:] = Rb
print dummy2.shape


K = Sn1 + np.dot(dummy1,np.transpose(dummy2))
print K.shape


Up = np.zeros((15,7),dtype=np.float64)
Up[:,:3] = Un
Up[:,3:] = P
print Up.shape


Vq = np.zeros((10,7),dtype=np.float64)
Vq[:,:3] = np.transpose(Vn)
Vq[:,3:] = Q
print Vq.shape


BigMat = np.dot(Up,np.dot(K,np.transpose(Vq)))
print "Rank", np.linalg.matrix_rank(BigMat)


#Test whether our code is correct or not
print BigMat
print a + np.dot(A,np.transpose(B))
