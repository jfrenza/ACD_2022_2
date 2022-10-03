### TALLER 1: ALGEBRA PARA CIENCIAS DE LOS DATOS

## PUNTO 6

# Programe la Distancia de Malahanobis utilizando la covarianza habitual,
# luego la covarianza bajo el shrinkage de Ledoit and Wolf (cov1para.m)
# y la covarianza y vector de medias robustas obtenida bajo el método de
# mínima curtosis (Kurmain.m). Ilustre ejemplos concretos donde el shrinkage y el método
# robusto presenta un mejor tendimiento y comente los resultados.

import numpy as np

data = np.array([
[40, 20, 15],
[35, 500, 10],
[45, 200, 20],
[60, 200, 10],
[90, 150, 45]])

x = np.array([15, 600, 50])

# Distancia de Mahalanobis utilizando la covarianza habitual

def MahalDistance(data, x):

    m = np.mean(data, axis = 0)
    data = np.transpose(data)
    Cov = np.cov(data)
    InvCov = np.linalg.inv(Cov)
    temp1 = np.dot((x - m), InvCov)
    temp2 = np.dot(temp1, np.transpose(x - m))
    MD = np.sqrt(temp2)
    return np.reshape(MD, -1)

print(MahalDistance(data, x))

# Distancia de Malahanobis utilizando Shinkage de Ledoit Wolf
