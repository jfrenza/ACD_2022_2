## PUNTO 13

# Calcule el número condición de la matriz de covarianzas de portfolio100 en su versión
#original. Implemente alternativas para mejorarlo y reducirlo a la cuarta parte.

import numpy as np
import pandas as pd

df = pd.read_excel('Portfolio.xlsx')

matrix = df.to_numpy()

CovM = np.cov(matrix)
SizeCov = CovM.shape[1]
I= np.identity(SizeCov)
CondicionOriginal =  np.linalg.cond(CovM)

for i in range(2,10):
    CovS = CovM + (i*I)
    CondicionNueva = np.linalg.cond(CovS)
    if 4 * CondicionNueva < CondicionOriginal:
        break

print(f'El número de condición de la matriz de Covarianzas original es: {CondicionOriginal.round(2)}')
print(f'El número de condición de la matriz de Covarianzas Nueva es: {CondicionNueva.round(2)} con Lambda igual a: {i}')
