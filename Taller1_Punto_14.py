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
from sklearn.covariance import LedoitWolf


# Simulación utilizando la matriz de covarianza tradicional

Condicion = []
Determinante = []

Size = 1000
for n in range(2,102):
    Hilbert = hilbert(n)
    mean = np.zeros(n)
    datos = np.random.multivariate_normal(mean, cov = Hilbert, size = (Size))
    CovariazaEstimada = np.cov(datos)
    CondicionMatriz =  np.linalg.cond(CovariazaEstimada)
    detMatriz =  np.linalg.det(CovariazaEstimada)
    Condicion.append(CondicionMatriz)
    Determinante.append(detMatriz)


fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('N Vs Número de Condición & N Vs Determinante')
ax1.plot(range(100), Condicion)
ax2.plot(range(100), Determinante)
plt.show()

# Simulación utilizando la matriz de covarianza bajo el Shrinkage de Ledoit and Wolf

Condicion2 = []
Determinante2 = []

N = 1000
for n in range(2,102):
    Hilbert2 = hilbert(n)
    media = np.zeros(n)
    simulacion = np.random.multivariate_normal(media, cov = Hilbert2, size = N)
    cov = LedoitWolf().fit(simulacion)
    CovM = cov.covariance_
    CondicionMatriz2 =  np.linalg.cond(CovM)
    detMatriz2 =  np.linalg.det(CovM)
    Condicion2.append(CondicionMatriz2)
    Determinante2.append(detMatriz2)


fig, (ax3, ax4) = plt.subplots(2)
fig.suptitle('N Vs Número de Condición & N Vs Determinante (Ledoit & Wolf)')
ax3.plot(range(100), Condicion2)
ax4.plot(range(100), Determinante2)
plt.show()
