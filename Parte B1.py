import numpy as np
import matplotlib.pyplot as plt

print("\nParte B1")

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


#El métrodo construye la ecuación matricial Va=Z a partir de los datos
#Cada punto de la función corresponde a la ecuación P(xi)=a0+a1x0+a2x0^2+...+anx0^n=y0
#Escribiendo cada ecuación para cada punto, se obtiene la ecuación matricial donde V es la matriz de Vandermonde
V = np.vander(f, N=len(f), increasing=True)#Crea la matriz de Vandermonde correspondiente a los 30 datos
coef_matricial = np.linalg.solve(V, Z)#Resuelve la ecuación matricial Va=Z para a, esto corresponde a los coeficientes del polinomio interpolante

#La siguiente función evalua el polinomio creado por el método matricial
def evaluar_polinomio_matricial(f_eval):
    f_eval = np.asarray(f_eval, dtype=float)#Vuelve nuestra entrada en un arreglo
    resultado = np.zeros_like(f_eval, dtype=float)#Crea una arreglo de ceros del mismo tamaño que la entrada

    for k, ak in enumerate(coef_matricial):#Bucle que recorre cada coeficiente de la matriz de coeficientes obtenida
        resultado += ak * f_eval**k #Realiza la operación ak*f^k, donde ak es el coeficiente en la matriz obtenida y k el índice.
                                    #Para k=0, nótese que ak=a0 y por tanto, el primer resultado es a0*f^0=a0.
                                    #Luego, el bucle sigue para k=1, por lo que resultado=a0+a1*f^1(esto es el resultado anterior más el siguiente)+
                                    #Se sigue así para todos los 30 valores

    return resultado #La función devuelve el valor del polinomio evaluado en el punto que deseamos

Z_1000_matricial = evaluar_polinomio_matricial(np.array([1000.0]))[0]
#Esto devuelve el valor que deseamos evaluar. Dado que la función produce un arreglo, mediante el índice [0] indicamos que queremos el primer valor del arreglo, que corresponde al valor evaluado


print("\nMétodo matricial grado 29:")
print(f"|Z|(1000 Hz) = {Z_1000_matricial:.6f} ohm")

#La siguiente función evalua usando el polinomio de interpolación por el método de Lagrange
def lagrange_interpolation(f_eval, f_data, Z_data):
    f_eval = np.asarray(f_eval, dtype=float)#Vuelve a la entrada en un arreglo numérico
    Z_eval = np.zeros_like(f_eval, dtype=float)#Realiza lo mismo pero para el segundo parámetro

    n = len(f_data)#La cantidad de datos

    for i in range(n):#Bucle que recorre los 30 datos
        Li = np.ones_like(f_eval, dtype=float)#Crea un arreglo del mismo tamaño que la entrada cuyos elementos son todos 1

        for j in range(n):#Recorre nuevamente los 30 datos para el cálculo del Li
            if i != j:#Esto evita la división por 0, como exige el cálculo de Li para Lagrange
                Li *= (f_eval - f_data[j]) / (f_data[i] - f_data[j])#Es cálculo de los Li, multiplicando los x-xj/xi-xj

        Z_eval += Z_data[i] * Li#Calcula el resultado sumando los xi*Li

    return Z_eval

Z_1000_lagrange = lagrange_interpolation(np.array([1000.0]), f, Z)[0]#De la misma forma que en la parte anterior, se toma el índice 0
                                                                     #Porque es el índice que corresponde al valor de la evaluación buscada

print("\nMétodo de Lagrange grado 29:")
print(f"|Z|(1000 Hz) = {Z_1000_lagrange:.6f} ohm")

#Ahora se comparan usando 3 distintos grados de polinomios con un polinomio de grado 29(el que se obteniene de la interpolación)
grados = [5, 10, 15, 29] #Lista que contiene los grados
polinomios = {}#Diccionario que almacenará los polinomios

print("\nComparación de polinomios en f = 1000 Hz:")

for grado in grados:#Bucle que recorre la lista de grados
    coef = np.polyfit(f, Z, grado)#Crea un arreglo de coeficientes para un polinomio de grado indicado que ajusta a los datos. Esto para cada grado que recorre el bucle
    P = np.poly1d(coef)#Crea un polinomio con los coeficientes anteriores que ajusta a los datos cuya variable independiente es f y Z la dependiente
    polinomios[grado] = P #Esto guarda el polinomio obtenido anteriormente en el diccionario

    print(f"Grado {grado}: |Z|(1000 Hz) = {P(1000.0):.6f} ohm")


#Ahora se comparan los polinimos anteriormente obtenidos gráficamente
plt.figure(figsize=(10, 6))

plt.plot(f, Z, "ko", label="Datos experimentales")
#Las líneas son similares a la de la parte A, son los elementos de cada gráfica
for grado in [5, 10, 15]:#Recorre cada grado de los polinomios de ajuste obtenidos
    P = polinomios[grado]
    plt.plot(f, P(f), "o-", label=f"Polinomio grado {grado}")#Crea la gráfica de cada polinomio de ajuste

plt.xlabel("Frecuencia f [Hz]")
plt.ylabel("|Z| [ohm]")
plt.title("Comparación de polinomios evaluados solo en los datos originales")
plt.grid(True)
plt.legend()
plt.show()

#Ahora seleccionamos el de grado 10 para realizar la validación LOO
grado_seleccionado = 10 #Grado del polinomio seleccionado
P_seleccionado = polinomios[grado_seleccionado] #Selecciona el polinomio del grado que seleccionamos

Z_1000_polinomio = P_seleccionado(1000.0) #Evalúa el polinomio para Z=1000ohm en el polinomio seleccionado

print("\nPolinomio seleccionado:")
print(f"Se selecciona el polinomio de grado {grado_seleccionado}.")
print(f"|Z|(1000 Hz) = {Z_1000_polinomio:.6f} ohm")

#Ahora se realiza la validación LOO
np.random.seed(7)#Fija el comportamiento del generador aleatorio para las siguientes líneas de código, lo que se denomina semilla
                 #Esto es para que cada que ejecutemos el código, este seleccione los mismos 5 puntos aleatorios
                 #Así, permite reproducir muchas veces el algoritmo usando los mismos 5 datos
indices_loo = np.sort(np.random.choice(len(f), size=5, replace=False))#Crea un arreglo de 5 posiciones aleatorias y los ordena de menor a mayor

errores_relativos = []#Lista donde se guardarán los errores relativos

print("\nValidación Leave-One-Out usando 5 puntos:")

#Algoritmo que realiza la validación LOO
for idx in indices_loo:#Recorre cada índice en el arreglo aleatorio de índices
    f_train = np.delete(f, idx)#Elimina temporalmente el valor de f correspondiente al índice
    Z_train = np.delete(Z, idx)#Lo mismo, pero para Z, con ambos se elimina el punto en cuestión
    coef_temp = np.polyfit(f_train, Z_train, grado_seleccionado)#Realizamos el mismo procedimiento de arriba, creando el polinomio de ajuste
    P_temp = np.poly1d(coef_temp)
    Z_predicho = P_temp(f[idx])#Este es el valor de Z con el polinomio nuevo que usamos
    Z_real = Z[idx]#Tomamos el valor real de la lista para calcular el error relativo
    error_relativo = abs(Z_predicho - Z_real) / abs(Z_real) * 100#Calcula el error relativo

    errores_relativos.append(error_relativo)#Guarda cada error en la lista de errores

    print("--------------------------------")
    print(f"Punto eliminado:")
    print(f"f = {f[idx]:.2f} Hz")
    print(f"Z real = {Z_real:.6f} ohm")
    print(f"Z predicho = {Z_predicho:.6f} ohm")
    print(f"Error relativo = {error_relativo:.6f} %")

error_promedio = np.mean(errores_relativos)#Esto calcula la media de los errores relativos obtenidos

print("--------------------------------")
print(f"Error relativo promedio Leave-One-Out = {error_promedio:.6f} %")