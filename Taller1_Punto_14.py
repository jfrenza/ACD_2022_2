## PUNTO 14

#Sea Hn la matriz de Hilbert. Simule 1000 datos normales con matriz de covarianza
# Hn. Estime la matriz de covarianzas desde los datos simulados. Realice una gráfica
# de n en el eje x con el número de condición de la matriz de covarianza estimada
# en el eje y. Qué tipo de comportamiento observa. Haga lo mismo para su determinante.
# Realice lo mismo estimando la matriz de covarianza utilizando el shrinkage de Ledoit and Wolf.
# Compare los resultados. Haga un análisis gráfico y de visualización donde se
# observe si al final el shrinkage mejora el número condición.


import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot as plt


data = []

for n in range(2,102):
    size = 1000
    Hilbert = hilbert(n)
    mean = np.zeros(n)
    datos = np.random.multivariate_normal(mean, cov = Hilbert, size = (size))
    CovariazaEstimada = np.cov(datos)
    CondicionMatriz =  np.linalg.cond(CovariazaEstimada)
    data.append(CondicionMatriz)

#plt.plot(range(98), data)
#plt.show()


a = range(2,102)
print(len(a))
