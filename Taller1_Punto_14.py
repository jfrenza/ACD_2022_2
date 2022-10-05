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

n = 1000

H = hilbert(n)
mean = np.zeros(n)
datos = np.random.multivariate_normal(mean, cov = H, size = (n))

print(datos)

import matplotlib.pyplot as plt
plt.plot(datos[:, 0], datos[:, 1], '.', alpha=0.5)
plt.axis('equal')
plt.grid()
plt.show()
