import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [122,114,86,134,146,107,68,117,71,98],
    [139,126,90,144,163,136,61,62,41,120]
    ])

Y = np.array([0.115, 0.120, 0.105, 0.090, 0.100, 0.120, 0.105, 0.080, 0.100, 0.115])[np.newaxis]


def mount_X(X,n):
    mat = np.ones([X.shape[1],X.shape[0]])

    for loop in range(n):
        mat = np.concatenate((mat,X.T**(loop+1)),axis=1)
    return mat


GRAU = 3

for GRAU in range(1,10):
    mat_X = mount_X(X,GRAU)
    mat_Y = np.transpose(Y)
    mat_B = np.dot(np.dot(np.linalg.pinv(np.dot(mat_X.T,mat_X)),mat_X.T),mat_Y)

    y_chapeu = np.dot(mount_X(X, GRAU), mat_B)
    
    a = Y.T-y_chapeu
    print("GRAU = "+str(GRAU))
    print(
        "R² = "+str(
                                1-
                                (
                                    sum((Y.T-y_chapeu)**2)/
                                    (sum((Y.T - np.mean(Y))**2))
                                ))+
                    "\nR² ajustado = "+str(
                                1-
                                (
                                    (sum((Y.T-y_chapeu)**2)/((mat_X.shape[0]/2)-mat_X.shape[1]))/
                                    (sum((Y.T - np.mean(Y))**2)/((mat_X.shape[0]/2) - 1))
                                )                    
                            )
    )
    print("----------------------------")