## PUNTO 13

# Calcule el número condición de la matriz de covarianzas de portfolio100 en su versión
#original. Implemente alternativas para mejorarlo y reducirlo a la cuarta parte.

import numpy as np
import pandas as pd

# Se importan la librerías necesias para el código

df = pd.read_excel('Portfolio.xlsx') # Se hace uso de la librería pandas .read_excel() para cargar el dataset

matrix = df.to_numpy() # Se hace uso de la función .to_numpy() con el objetivo de transformar el dataset en
                        # una matriz.

CovM = np.cov(matrix) # Con la función np.cov() se obtiene la matriz de covarianza de los datos y se almacena
                        # en la variable llamada CovM

SizeCov = CovM.shape[1] # En la varibale  SizeCov está almacenado el tamaño de la matriz de covarianzas

I = np.identity(SizeCov) # La variable I es la matriz identidad de tamaño igual a la matriz de Covarianzas

CondicionOriginal =  np.linalg.cond(CovM) # Con el función np.linalg.cond() se calcula el número de condición de la
                                            # matriz de covarianzas y se almacena en la variable CondicionOriginal

i = 1

while True:
    i = i + 1
    CovS = CovM + (i * I)
    CondicionNueva = np.linalg.cond(CovS)
    if 4 * CondicionNueva < CondicionOriginal:
        break

# El ciclo indefinido anterior toma la matriz de covarianzas y le suma i-veces la matriz identidad.
# ese proceso incrementa la diagonal lo que ayuda a que se disminuya el número de condición.
# Una vez la condición nueva es al menos 4 veces inferior a la Original se rompe el ciclo.

print(f'El número de condición de la matriz de Covarianzas original es: {CondicionOriginal.round(2)}')
print(f'El número de condición de la matriz de Covarianzas Nueva es: {CondicionNueva.round(2)} con Lambda igual a: {i}')
