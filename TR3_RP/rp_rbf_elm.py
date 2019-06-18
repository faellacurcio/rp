import numpy
import matplotlib
import matplotlib.pyplot as plt
from rp_rbf_class import RBF


def fix_Y(vec_input):
    mat = numpy.zeros([vec_input.shape[0], int(numpy.amax(vec_input))])
    for idx, element in enumerate(Y):
        mat[idx][int(element)-1] = 1
    return mat


samples = list()
# Abre e lê dermatology como um dataset
with open('twomoons.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split())))
dataset = numpy.array(samples)

# separa os dados em features e classe (X/Y)
X = dataset[:, :-1]
Y = dataset[:, dataset.shape[1]-1]

# Função de normalização


def zscore(X_vector):
    X_vector = X_vector - numpy.mean(X_vector, axis=0)
    X_vector = X_vector / numpy.std(X_vector, axis=0, ddof=1)
    return X_vector


# X = zscore(X)


OBJ_RBF = RBF(15)

# classifica as amostras da base de teste
OBJ_RBF.fit(X.T, Y.T)

min_features = numpy.amin(X, axis=0)
max_features = numpy.amax(X, axis=0)
linalg_var = numpy.linspace(min_features, max_features, 100)
X_test = []

for val_min in linalg_var[:, 0]:
    for val_max in linalg_var[:, 1]:
        X_test.append([val_min, val_max])
X_test = numpy.array(X_test)

y = OBJ_RBF.predict(X_test.T)
#                    blue                        red
colors = numpy.array(98*(y > 0)).T + numpy.array(114*(y < 0)).T
colors = [chr(x) for x in colors]

plt.scatter(X_test[:, 0], X_test[:, 1], c=colors)

#                     green                       yellow
colors = numpy.array(103*(Y > 0)).T + numpy.array(121*(Y < 0)).T
plt.scatter(X[:, 0], X[:, 1], c=colors)

plt.title("superfície de decisão two moons")

plt.savefig('15 neuronios.png')

matplotlib.pyplot.clf()

print('Saving image: 15 neuronios.png')
