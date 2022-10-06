### TALLER 1: ALGEBRA PARA CIENCIAS DE LOS DATOS

## PUNTO 6

# Programe la Distancia de Malahanobis utilizando la covarianza habitual,
# luego la covarianza bajo el shrinkage de Ledoit and Wolf (cov1para.m)
# y la covarianza y vector de medias robustas obtenida bajo el método de
# mínima curtosis (Kurmain.m). Ilustre ejemplos concretos donde el shrinkage y el método
# robusto presenta un mejor rendimiento y comente los resultados.

import numpy as np
from sklearn.covariance import LedoitWolf

# Se importan las librerias necesarias para el código

data = np.array([
[40, 20, 15],
[35, 500, 10],
[45, 200, 20],
[60, 200, 10],
[90, 150, 45]])

# data es una matriz de datos

x = np.array([15, 600, 50]) # x es el vector al cuál se calculará la distancia de el mismo hasta los datos

# Distancia de Mahalanobis utilizando la covarianza habitual

def MahalDistance(data, x):

    m = np.mean(data, axis = 0) # m es el vector de medias de cada una de las columnas de datos
    data = np.transpose(data) # En esta línea se transpone la matriz para que coincida la dimensión con el vector x
    Cov = np.cov(data) # se hace uso de la función .cov() de Numpy para encontrar la matriz de covarianzas de los datos
    InvCov = np.linalg.inv(Cov) # se hace uso de la función .linalg.inv() de Numpy para hallar la inversa de la matriz de covarianzas
    temp1 = np.dot((x - m), InvCov) # En esta primera variable temporal se almacena la multiplicación entre el vector x menos vector m & la matriz inversa
    temp2 = np.dot(temp1, np.transpose(x - m)) # En la variable teporal2 está almacenada la multiplicación entre la varibale temp1 y la diferencia  de x y m transpuesta
    MD = np.sqrt(temp2) # esta linea obtiene la raíz cuadrada de la variable temporal 2
    return np.reshape(MD, -1) # en esta línea se regresa el valor de la distancia de Mahalanobis

# Distancia de Malahanobis utilizando Shinkage de Ledoit Wolf

def MahalLedoitWolf(data, x):
    m = np.mean(data, axis = 0)
    cov = LedoitWolf().fit(data)
    CovLedoitWolf = cov.covariance_
    InvCov = np.linalg.inv(CovLedoitWolf)
    temp1 = np.dot((x - m), InvCov)
    temp2 = np.dot(temp1, np.transpose(x - m))
    MD = np.sqrt(temp2)
    return np.reshape(MD, -1)

# La función MahalLedoitWolf tiene exactamente los mismos pasos que la función MahalDistance; con la diferencia de que
# el cálculo de la matriz de covarianzas se hace a través del método de Ledoit and Wolf. Para ello se hace importa LedoitWolf()
# y se utiliza la función .fit() con los datos. Eso crea un objeto LedoitWolf() al que posteriormente se le calcula la matriz
# de covarianzas. Haciendo uso de .covariance_

# Distancia de Mahalanobis utilizando el método de mínima Kurtosis

print(MahalDistance(data, x))
print(MahalLedoitWolf(data, x))
