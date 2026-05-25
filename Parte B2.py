import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

print("\nParte B2")

f = np.array([
    100, 120, 145, 170, 200, 235, 270, 310, 355, 405,
    460, 520, 585, 655, 730, 810, 895, 985, 1080, 1180,
    1290, 1410, 1540, 1680, 1830, 1990, 2160, 2340, 2530, 2730
], dtype=float)

Z = np.array([
    152.3, 149.1, 146.8, 144.9, 142.0, 139.5, 137.9, 136.1, 134.8, 133.6,
    132.7, 131.9, 131.4, 131.1, 130.9, 131.0, 131.3, 131.9, 132.7, 133.8,
    135.2, 136.9, 138.9, 141.1, 143.5, 146.1, 149.0, 152.2, 155.6, 159.2
], dtype=float)

grado_seleccionado=10
coef_polinomio = np.polyfit(f, Z, grado_seleccionado)
P_seleccionado = np.poly1d(coef_polinomio)

#Ahora construiremos un spline cúbico natural para los 30 puntos
#Natural implica que que, además de las condiciones para un spline, la segunda derivada en los extremos sea 0
#Esto asegura que en los extremos, el spline no tenga una curvatura exagerada en los extremos
#la linea from scipy.interpolate import cubicspline importa la función para crear spiline cubicos de la librería scypi
#Esto permite crear spline cúbicos en Python

spline = CubicSpline(f, Z, bc_type="natural")#Crea el spline cúbico con los datos obtenidos. el parámetro bc_type=natural indica que es de tipo natural

Z_1000_spline = spline(1000.0)#Calcula el valor de la impedancia con el spline para f=1000

print("Spline cúbico natural:")
print(f"|Z|(1000 Hz) = {Z_1000_spline:.6f} ohm")


Z_1000_polinomio = P_seleccionado(1000.0)#Calcula el valor usando el polinomio de ajuste de grado 10 para comparar

print("\nComparación en f = 1000 Hz:")
print(f"Polinomio grado {grado_seleccionado}: |Z|(1000 Hz) = {Z_1000_polinomio:.6f} ohm")
print(f"Spline cúbico natural:               |Z|(1000 Hz) = {Z_1000_spline:.6f} ohm")
print(f"Diferencia absoluta = {abs(Z_1000_spline - Z_1000_polinomio):.6f} ohm")

#Ahora se grafica el polinomio, spline y los datos
#Dado que queremos observar una curva suave, es necesario graficar datos intermedios en la frecuencia que queremos buscar
#Esto se debe a que experimentalmente obtuvimos 30 datos, pero esto no es suficiente para visualizar una curva suave
#Para esto, calcularemos muchos valores intermedios en la frecuencia para poder dibujar mejor la curva del spline
f_fino = np.linspace(f.min(), f.max(), 1000)#Crea un arreglo de 1000 valores entre la frecuencia mínima y la máxima de los datos

Z_spline_fino = spline(f_fino)#Crea el spline utilizando la malla fina creada anteriormente
Z_polinomio_fino = P_seleccionado(f_fino)#Crea el polinomio de ajuste de grado 10 con la malla fina que creamos

plt.figure(figsize=(10, 6))

plt.plot(f, Z, "ko", label="Datos experimentales")
plt.plot(f_fino, Z_spline_fino, label="Spline cúbico natural")
plt.plot(f_fino, Z_polinomio_fino, "--", label=f"Polinomio grado {grado_seleccionado}")

plt.xlabel("Frecuencia f [Hz]")
plt.ylabel("|Z| [ohm]")
plt.title("Parte B2: Spline cúbico natural vs polinomio seleccionado")
plt.grid(True)
plt.legend()
plt.show()
#Esto crea y muestra la gráfica que queremos ver