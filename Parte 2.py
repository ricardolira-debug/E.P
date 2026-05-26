import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
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

print("\nParte 2")
#En esta parte usaremos diferenciación numérica para encontrar dV/df
#En este ejercicio, los datos están igualmente espaciados, por lo que nuestro h será la diferencia entre cualquier par de frecuencias consecutivas
h = f[1] - f[0]
print(f"Espaciamiento h = {h:.2f} kHz")

#Ahora, necesitamos una función que nos de el índice de una determinada frecuencia que querramos del arreglo de frecuencias
def indice_frecuencia(f_data, f_objetivo):
    indices = np.where(np.isclose(f_data, f_objetivo))[0]#Busca la posición de una frecuencia dada. Se usa np.isclose por si la frecuencia
                                                         #Buscada es númericamente cercana a una que está en la lista(si hubiera datos como 39.999999) 
                                                         #Todo esto lo almacena en una tupla                                                      
    if len(indices) == 0:#Esto comprueba si la frecuencia que se usa está en la tabla
        raise ValueError(f"La frecuencia {f_objetivo} kHz no está en la tabla.")

    return indices[0]#Devuelve la posición de la frecuencia buscada
#Ahora se usa una función para realizar la diferenciación numérica centrada de orden 2
def derivada_centrada_orden2(f_data, V_data, f_objetivo):
    i = indice_frecuencia(f_data, f_objetivo)#Busca el índice de la frecuencia buscada
    h = f_data[1] - f_data[0]#El valor de h

    if i - 1 < 0 or i + 1 >= len(f_data):#Esto comprueba si es que hay un punto anterior. y posterior. Si no hay, no se puede aplicar esta diferenciación
        raise ValueError("No hay puntos suficientes para diferencia centrada de orden 2.")

    derivada = (V_data[i + 1] - V_data[i - 1]) / (2 * h)#Calcula el valor de la derivada en el punto dado

    return derivada#Retorna el valor de la derivada
#Ahora con la derivada centrada de cuarto orden
def derivada_centrada_orden4(f_data, V_data, f_objetivo):#Las líneas son similares, la diferencia es que seta usa 4 puntos a diferencia de la anterio que usa 2
    i = indice_frecuencia(f_data, f_objetivo)
    h = f_data[1] - f_data[0]
    if i - 2 < 0 or i + 2 >= len(f_data):
        raise ValueError("No hay puntos suficientes para diferencia centrada de orden 4.")
    derivada = (
        -V_data[i + 2]
        + 8 * V_data[i + 1]
        - 8 * V_data[i - 1]
        + V_data[i - 2]
    ) / (12 * h)
    return derivada
#Si es que no hubiera la posibilidad de usar la derivación centrada, usamos la derivación progresiva
def derivada_progresiva_orden2(f_data, V_data, f_objetivo):#La lógica es la misma, solo cambia la fórmula para la derivada
    i = indice_frecuencia(f_data, f_objetivo)
    h = f_data[1] - f_data[0]

    if i + 2 >= len(f_data):
        raise ValueError("No hay puntos suficientes para fórmula progresiva de orden 2.")

    derivada = (
        -3 * V_data[i]
        + 4 * V_data[i + 1]
        - V_data[i + 2]
    ) / (2 * h)

    return derivada
#Ahora calculamos la derivada usando el spline y la función derivative de Numpy
spline_V = CubicSpline(f, V, bc_type="natural")
spline_dV = spline_V.derivative(1)

frecuencias_derivada = [40.0, 70.0, 100.0]
#Ahora mostramos los resultados
print("\nDerivadas en puntos interiores:")

for f_obj in frecuencias_derivada:
    dV_orden2 = derivada_centrada_orden2(f, V, f_obj)
    dV_orden4 = derivada_centrada_orden4(f, V, f_obj)
    dV_spline = spline_dV(f_obj)

    print("--------------------------------")
    print(f"f = {f_obj:.1f} kHz")
    print(f"Diferencia centrada orden 2 = {dV_orden2:.6f} V/kHz")
    print(f"Diferencia centrada orden 4 = {dV_orden4:.6f} V/kHz")
    print(f"Derivada del spline          = {dV_spline:.6f} V/kHz")
#Dado que 10 es un extremo, usamos la derivada progresiva
dV_10_prog = derivada_progresiva_orden2(f, V, 10.0)
dV_10_spline = spline_dV(10.0)

print("\nDerivada en el extremo inferior:")
print("--------------------------------")
print("f = 10.0 kHz")
print(f"Fórmula progresiva orden 2 = {dV_10_prog:.6f} V/kHz")
print(f"Derivada del spline        = {dV_10_spline:.6f} V/kHz")
