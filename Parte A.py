"""""Parte A"""
import numpy as np
import matplotlib.pyplot as plt

f = np.array([
    100, 120, 145, 170, 200, 235, 270, 310, 355, 405,
    460, 520, 585, 655, 730, 810, 895, 985, 1080, 1180,
    1290, 1410, 1540, 1680, 1830, 1990, 2160, 2340, 2530, 2730
], dtype=float)#Se crea un arreglo de las frecuencias

Z = np.array([
    152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6,
    132.7, 131.9, 131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 132.7, 133.8,
    135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 152.2, 155.6, 159.2
], dtype=float)#Se crea un arreglo de las impedancias



plt.figure(figsize=(9, 5))

plt.plot(f, Z, "o-", label="Datos experimentales")

plt.xlabel("Frecuencia f [Hz]")
plt.ylabel("Magnitud de impedancia |Z| [ohm]")
plt.title("Parte A: Impedancia en función de la frecuencia")

plt.grid(True)
plt.legend()
plt.show()
#Son los elementos que se muestran en la gráfica. Se indica qué valores se grafican, los títulos de los ejes, muestra de la cuadrícula, etc
#plt.show() muestra la gráfica
indice_minimo = np.argmin(Z)#Determina la posición del valor mínimo en el arreglo de impedancias
#Dado que los valores fueron introducidos par a par, entonces el valor de la posición de z_mínimo dará la frecuencia que le corresponde
f_minimo = f[indice_minimo]#Valor de la frecuencia en la posición de impedancia mínima
Z_minimo = Z[indice_minimo]#Valor de impedancia mínima

print("Mínimo visual aproximado usando los datos:")
print(f"Frecuencia aproximada del mínimo: {f_minimo:.2f} Hz")
print(f"Impedancia mínima aproximada: {Z_minimo:.4f} ohm")