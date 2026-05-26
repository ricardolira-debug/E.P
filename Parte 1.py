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

Z = np.array([
    182.4, 178.9, 175.1, 171.0, 166.8, 162.7, 158.9, 155.4, 152.0, 149.0,
    146.1, 145.2, 145.8, 147.3, 149.9, 153.5, 158.0, 163.2, 168.9, 174.8,
    180.5, 186.2, 191.5, 196.2, 200.1, 203.1, 205.2, 206.3, 206.1, 204.7,
    198.0, 194.4, 190.9, 187.8, 185.1, 183.0, 181.6, 180.8, 180.6, 180.9
], dtype=float)
#Cada uno de estos arreglos contiene los datos en orden de ternas, por ejemplo, según la tabla, f=10 cuando v=0.842 y z=182.4.
#Es importante que sea así como en los casos anteriores para poder crear las funciones requeridas
#Algoritmo del polinomio de Lagrange
def lagrange_interpolation(f_eval, f_data, y_data):#En este caso, ya que vamos a interpolar 2 cosas distintas, asignamos como segundo parámetro y_data
    f_eval = np.asarray(f_eval, dtype=float)#Vuelve a la entrada en un arreglo numérico
    y_eval = np.zeros_like(f_eval, dtype=float)#Realiza lo mismo pero para el segundo parámetro

    n = len(f_data)#La cantidad de datos

    for i in range(n):#Bucle que recorre los datos
        Li = np.ones_like(f_eval, dtype=float)#Crea un arreglo del mismo tamaño que la entrada cuyos elementos son todos 1

        for j in range(n):#Recorre nuevamente los 30 datos para el cálculo del Li
            if i != j:#Esto evita la división por 0, como exige el cálculo de Li para Lagrange
                Li *= (f_eval - f_data[j]) / (f_data[i] - f_data[j])#Es cálculo de los Li, multiplicando los x-xj/xi-xj

        y_eval += y_data[i] * Li#Calcula el resultado sumando los xi*Li

    return y_eval
#Ahora bien, se pide usar un polinomio de Lagrange de grado 2, por lo que queremos buscar 3 puntos para que el polinomio sea de grado
#Estos puntos deben ser lo más cercano posible a los puntos que deseamos estimar, por lo que haremos que se busquen los puntos más cercanos
#Con la siguiente función
def puntos_cercanos(f_data, y_data, f_evaluado):
    distancia=np.abs(f_data-f_evaluado)#Esto calcula la distancia entre cada frecuencia de la tabla y el dato que queremos estimar
    posicion=np.argsort(distancia)[:3]#Esta función toma las posiciones de las distancias y las ordena de menor a mayor distancia. El [:3] hace que se tomen solo las 3 primeras posiciones
                                      #Como están ordenadas de menor a mayor, las 3 primeras son las 3 más cercanas
    posicion=np.sort(posicion)#La línea anterior devuelve las distancias en orden ascendente en función a la distancia, esta las ordena de menor a mayor
    return f_data[posicion], y_data[posicion]#Devuelve los valores más cercanos de la frecuencia más cercana
                                             #También devuelve sus 3 valores correspondientes del segundo parámetro(voltaje o impedancia
spline_V=CubicSpline(f, V, bc_type="natural")#Crea los splines cúbicos naturales para V y Z
spline_Z=CubicSpline(f, Z, bc_type="natural")
frecuencias_pedidas=[41.0,73.0]#Son las frecuencias que se van a estimar para cada dato
for f_eval in frecuencias_pedidas:#Recorre cada frecuencia en la lista que se pedía
    f_V_cercanos, V_cercanos=puntos_cercanos(f, V, f_eval)#Usa la función de los puntos para obtener los puntos más cercanos para f y V
    V_spline=spline_V(f_eval)#Calcula el voltaje estimado usando el spline cúbico
    V_lagrange=lagrange_interpolation(f_eval, f_V_cercanos, V_cercanos)#Calcula el voltaje usando lagrange con los datos más cercanos
    #ahora hacemos lo mismo para Z
    f_Z_cercanos, Z_cercanos=puntos_cercanos(f, Z, f_eval)#Calcula los puntos más cercanos para f y Z
    Z_spline=spline_Z(f_eval)#Spline para Z
    Z_lagrange=lagrange_interpolation(f_eval, f_V_cercanos, Z_cercanos)
    print(f"Frecuencia evaluada: {f_eval:.1f}kHz")
    print("\nPuntos usados para Lagrange en V")
    for fi,vi in zip(f_V_cercanos, V_cercanos):
        print(f"frecuencia={fi:.1f}kHz, V={vi:.3f} V")
    print("\nPuntos usados para Lagrange en Z:")
    for fi,zi in zip(f_Z_cercanos, Z_cercanos):
        print(f"f={fi:.1f} kHz, Z={zi:.1f} ohm")
    print("\nResultados")
    print(f"V({f_eval:.1f}) por Lagrange = {V_lagrange:.6f}V")
    print(f"V({f_eval:.1f}) por spline cúbico = {V_spline:.6f}V")

    print(f"Z({f_eval:.1f}) por Lagrange = {Z_lagrange:.6f} ohm")
    print(f"Z({f_eval:.1f}) por spline cúbico = {Z_spline:.6f} ohm")
    #estas lineas imprimen los resultados, los valores de la frecuencia y los correspondientes de voltaje e impedancia por Lagrange y spline
    #Se aproximan a 1, 3 y 6 cifras por la resolución de los instrumentos mencionada en el ejercicio
#Ahora para comparar, usaremos las funciones plot de la librería matplotlib.pyplot
#Se usan las líneas similares para las preguntas del ejercicio 1
f_fino = np.linspace(f.min(), f.max(), 1000)#Malla fina como en el ejercicio anterior para el spline

plt.figure(figsize=(10, 6))
plt.plot(f, V, "ko", label="Datos V")
plt.plot(f_fino, spline_V(f_fino), label="Spline cúbico natural de V")
#

for f_eval in frecuencias_pedidas:
    plt.plot(f_eval, spline_V(f_eval), "ro")

plt.xlabel("Frecuencia f [kHz]")
plt.ylabel("Voltaje V [V]")
plt.title("Parte 1: Spline cúbico natural para V(f)")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(f, Z, "ko", label="Datos |Z|")
plt.plot(f_fino, spline_Z(f_fino), label="Spline cúbico natural de |Z|")

for f_eval in frecuencias_pedidas:
    plt.plot(f_eval, spline_Z(f_eval), "ro")

plt.xlabel("Frecuencia f [kHz]")
plt.ylabel("|Z| [ohm]")
plt.title("Parte 1: Spline cúbico natural para |Z|(f)")
plt.grid(True)
plt.legend()
plt.show()
#Son los elementos de la gráfica de V y Z