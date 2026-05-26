import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
f = np.array([
    10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5,
    35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5,
    60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0, 77.5, 80.0, 82.5,
    85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5
], dtype=float)
V = np.array([
    0.842, 0.911, 0.986, 1.062, 1.143, 1.227, 1.314, 1.401, 1.482, 1.551,
    1.216, 1.048, 0.866, 0.689, 0.521, 0.364, 0.223, 0.103, 0.012, -0.041,
    -0.057, -0.034, 0.018, 0.096, 0.197, 0.318, 0.452, 0.579, 0.700, 0.809,
    0.611, 0.688, 0.756, 0.811, 0.856, 0.894, 0.926, 0.954, 0.980, 1.004
], dtype=float)
print("\nParte 3")
#El ejercicio pide identificar los cambios de signo, para esto definimos la siguiente función
intervalos_raiz = []#Aquí se guardan los intervalos

for i in range(len(f) - 1):#Bucle que recorre todo el arreglo de datos de frecuencia
    if V[i] == 0:#Comprueba si el elemento es una raíz
        intervalos_raiz.append((f[i], f[i]))#Añade dicho elemento como un intervalo de extremos iguales, para indicar que es solo 1 elemento
    elif V[i] * V[i + 1] < 0:#Esto comprueba el cambio de signo
        intervalos_raiz.append((f[i], f[i + 1]))#Si lo hace, añade las raíces a la lista
print("\nIntervalos donde V(f) cambia de signo:")
for a, b in intervalos_raiz:
    print(f"[{a:.1f}, {b:.1f}] kHz")#Esto imprime los intervalos encontrados

#Para aplicar el método de la bisección, necesitamos una función continua que pase por los puntos indicados
#Para esto, usaremos el spline que hallamos anteriormente
spline_V=CubicSpline(f, V, bc_type="natural")
#ahora el algoritmo de la bisección, es el mismo que el del ejercicio anterior, con una toleancia de 10^-8 y 100 iteraciones
def biseccion(func, a, b, tol=1e-8, max_iter=100):

    fa = func(a)
    fb = func(b)
    if fa * fb > 0:
        raise ValueError("No hay cambio de signo en el intervalo.")
    for k in range(max_iter):
        c = (a + b) / 2
        fc = func(c)
        if abs(fc) < tol or abs(b - a) / 2 < tol:
            return c, k + 1
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, max_iter
#Ahora calculamos las raíces usando la bisección
raices_biseccion = []

print("\nRaíces usando bisección con interpolación lineal local:")

for a, b in intervalos_raiz:#Recorre los elementos de los intervalos hallados para la raíz
    raiz, iteraciones = biseccion(spline_V, a, b)#Calcula las raíces con el método de la bisección
    raices_biseccion.append(raiz)#Los añade a la lista de raíces
    print(f"Intervalo: [{a:.1f}, {b:.1f}] kHz")#Se imprimen los resultados
    print(f"Raíz aproximada = {raiz:.6f} kHz")
    print(f"Iteraciones = {iteraciones}")
    print(f"V(raíz) ≈ {spline_V(raiz):.6e} V")
#Ahora podemos encontrar las raíces usando el spline para comprar
# ============================================================
# COMPARACIÓN CON brentq SOBRE EL MISMO SPLINE
# ============================================================

raices_spline = []

print("\nComparación entre bisección y brentq usando el spline:")

for i, (a, b) in enumerate(intervalos_raiz):#Las líneas son similaresa a las del ejercicio anterior para calcular raíces con brentq
    raiz_brentq = brentq(spline_V, a, b)
    raices_spline.append(raiz_brentq)
    diferencia = abs(raices_spline[i] - raices_biseccion[i])
    print(f"Cruce {i + 1}")
    print(f"Bisección = {raices_biseccion[i]:.6f} kHz")
    print(f"brentq    = {raices_spline[i]:.6f} kHz")
    print(f"Diferencia = {diferencia:.6e} kHz")

#ahora comparamos los resultados
print("\nComparación bisección lineal vs spline:")

for i in range(len(raices_biseccion)):#Recorre la lista de raíces
    diferencia = abs(raices_spline[i] - raices_biseccion[i])#Calcula el error absoluto entre las raíces del spline y la bisección
    print(f"Cruce {i + 1}")#Muestra el cruce por
    print(f"Bisección lineal = {raices_biseccion[i]:.6f} kHz")
    print(f"Spline cúbico    = {raices_spline[i]:.6f} kHz")
    print(f"Diferencia       = {diferencia:.6f} kHz")