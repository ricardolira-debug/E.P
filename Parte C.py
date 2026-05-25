import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
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

#Además de importar la función CubicSpline, se importa la función brentq, que encuentra las raíces de una función continua
print("\nParte C")

spline = CubicSpline(f, Z, bc_type="natural")#Creamos el spline natural
spline_d1 = spline.derivative(1)#Calcula la primera derivada del spline
spline_d2 = spline.derivative(2)#Calcula la segunda derivada del spline

dZ_df = spline_d1(f)#Evalua la primera derivada del spline para cada dato

print("\nPrimera derivada en los puntos medidos:")

for fi, derivada in zip(f, dZ_df):#Bucle que recorre cada dato con su evaluación en la primera derivada
    print(f"f = {fi:8.2f} Hz    d|Z|/df = {derivada: .8f} ohm/Hz")#Imprime la primera derivada evaluada en cada dato


#Gráfica de la priemra derivada
plt.figure(figsize=(10, 6))

plt.plot(f, dZ_df, "o-", label="Primera derivada en los puntos")
plt.axhline(0, color="black", linestyle="--", label="d|Z|/df = 0")

plt.xlabel("Frecuencia f [Hz]")
plt.ylabel("d|Z|/df [ohm/Hz]")
plt.title("Parte C: Primera derivada usando spline cúbico")
plt.grid(True)
plt.legend()
plt.show()

#Ahora se busca el mínimo 
raices_derivada = []#Lista donde se almacena donde la primera derivada es 0

for a, b in zip(f[:-1], f[1:]):#Recorre una lista formada por cada subintervalo que hay en en la lista de datos
    if spline_d1(a) * spline_d1(b) < 0:#Busca el cambio de signo en los subintervalos
        raiz = brentq(spline_d1, a, b)#Calcula el mínimo mediante la función brentq
        raices_derivada.append(raiz)#Añade a la lista de raíces

print("\nPuntos donde d|Z|/df = 0:")

#Este algoritmo comprueba si cada punto encontrado es un máximo o un mínimo
for r in raices_derivada:
    Z_min = spline(r)
    segunda_derivada = spline_d2(r)

    print("--------------------------------")
    print(f"Frecuencia del extremo: {r:.6f} Hz")
    print(f"|Z| en ese punto:       {Z_min:.6f} ohm")
    print(f"Segunda derivada:       {segunda_derivada:.10f} ohm/Hz^2")

    if segunda_derivada > 0:
        print("Conclusión: es un mínimo local.")
    elif segunda_derivada < 0:
        print("Conclusión: es un máximo local.")
    else:
        print("Conclusión: la segunda derivada es aproximadamente cero.")

#Gráfica del spline 

f_fino = np.linspace(f.min(), f.max(), 1000)
Z_spline_fino = spline(f_fino)

plt.figure(figsize=(10, 6))

plt.plot(f, Z, "ko", label="Datos experimentales")
plt.plot(f_fino, Z_spline_fino, label="Spline cúbico natural")

for r in raices_derivada:
    plt.plot(r, spline(r), "ro", label="Mínimo encontrado")

plt.xlabel("Frecuencia f [Hz]")
plt.ylabel("|Z| [ohm]")
plt.title("Parte C: Mínimo encontrado usando derivada del spline")
plt.grid(True)
plt.legend()
plt.show()
