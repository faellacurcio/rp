import numpy as np
import matplotlib.pyplot as plt

def mount_X(X,n):
    mat = np.ones([X.shape[1],X.shape[0]])

    for loop in range(n):
        mat = np.concatenate((mat,X.T**(loop+1)),axis=1)
    return mat


samples = list()

# Abre e lê aerogerador como um dataset
with open('aerogerador.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split())))
dataset = np.array(samples)

# separa os dados em features e classe (X/Y) 
X = dataset[:,0][np.newaxis]
Y = dataset[:,1][np.newaxis]
GRAU = 3

for GRAU in range(2,7):
    mat_X = mount_X(X,GRAU)
    mat_Y = np.transpose(Y)
    mat_B = np.dot(np.dot(np.linalg.pinv(np.dot(mat_X.T,mat_X)),mat_X.T),mat_Y)

    y_chapeu = np.dot(mount_X(X, GRAU), mat_B)
    
    plt.scatter(X,Y)
    x_ = np.array(list(range(15)))

    plt.plot(X.T, y_chapeu,'r')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title("GRAU #"+str(GRAU))
    
    a = Y.T-y_chapeu
    plt.text(0,400,"R² = "+str(
                                1-
                                (
                                    sum((Y.T-y_chapeu)**2)/
                                    (sum((Y.T - np.mean(Y))**2))
                                ))+
                    "\nR² ajustado = "+str(
                                1-
                                (
                                    (sum((Y.T-y_chapeu)**2)/(mat_X.shape[0]-mat_X.shape[1]))/
                                    (sum((Y.T - np.mean(Y))**2)/(mat_X.shape[0] - 1))
                                )                    
                            ), bbox=dict(alpha=0.5))
    
    plt.savefig("q1_fig_GRAU"+str(GRAU))
    plt.clf()