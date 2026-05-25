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

print("\nParte D")

spline = CubicSpline(f, Z, bc_type="natural")#Creamos el spline cúbico natural
spline_d1 = spline.derivative(1)#Calculamos la primera derivada del spline
Z_th = 150.0#Esto servirá para definir la función Z(f)-z_th
def g(x):#se define la función Z(f)-Z_th
    return spline(x) - Z_th
raices = []#Lista para buscar las raíces
for a, b in zip(f[:-1], f[1:]):#Bucle que recorre una lista con los subintervalos obtenidos de los datos
    if g(a) == 0:#comprueba si algún extremo es raíz
        raices.append(a)#Lo añade a la lista si es que es así
    elif g(a) * g(b) < 0:#En caso contrario, busca si hay un cambio de signo
        raiz = brentq(g, a, b)#Si es así, calcula la raíz usando la función brentq y la añade a la lista
        raices.append(raiz)

print("\nRaíces encontradas de |Z|(f) = 150 ohm:")

for r in raices:#Este bucle imprime las raíces
    print("--------------------------------")
    print(f"f = {r:.4f} Hz")
    print(f"|Z|(f) = {spline(r):.4f} ohm")
    print(f"g(f) = {g(r):.4e}")
#Gráfica de la función con una malla fina

f_fino = np.linspace(f.min(), f.max(), 1000)
Z_spline_fino = spline(f_fino)
plt.figure(figsize=(10, 6))
plt.plot(f, Z, "ko", label="Datos experimentales")
plt.plot(f_fino, Z_spline_fino, label="Spline cúbico natural")
plt.axhline(Z_th, color="red", linestyle="--", label="Umbral 150 ohm")
for r in raices:
    plt.plot(r, spline(r), "ro")
plt.xlabel("Frecuencia f [Hz]")
plt.ylabel("|Z| [ohm]")
plt.title("Parte D: Raíces de |Z|(f) = 150 ohm")
plt.grid(True)
plt.legend()
plt.show()

#La siguiente función aplica el método de la bisección considerando un máximo de 100 iteraciones y una tolerancia de 0.0000001
def biseccion(func, a, b, tol=1e-8, max_iter=100):
    fa = func(a)
    fb = func(b)
    if fa * fb > 0:#Comprueba el cambio de signo. En caso no lo sea, imprime el mensaje
        raise ValueError("No hay cambio de signo en el intervalo.")
    for k in range(max_iter):#Bucle que realiza las 100 iteraciones para cada intervalo
        c = (a + b) / 2#Calcula el punto medio
        fc = func(c)#Calcula el valor de c
        if abs(fc) < tol or abs(b - a) / 2 < tol:#Si este valor cumple la tolerancia indicada, se detiene y retorna la raíz
            return c, k + 1
        if fa * fc < 0:#Si encuentra el cambio de signo con el extremo izquierdo, hace del nuevo extremo derecho el valor de c
            b = c
            fb = fc
        else:#Análogamente, si encuentra el cambio con el extremo derecho, hace del nuevo extremo derecho a c
            a = c
            fa = fc

    return c, max_iter

# Intervalos donde hay cambio de signo:
# raíz baja: entre 100 y 120
# raíz alta: entre 2160 y 2340

r1_bis, it1_bis = biseccion(g, 100, 120)
r2_bis, it2_bis = biseccion(g, 2160, 2340)

print("\nMétodo de bisección:")
print("--------------------------------")
print(f"Raíz baja = {r1_bis:.4f} Hz")
print(f"Iteraciones = {it1_bis}")
print(f"|Z| = {spline(r1_bis):.4f} ohm")

print("--------------------------------")
print(f"Raíz alta = {r2_bis:.4f} Hz")
print(f"Iteraciones = {it2_bis}")
print(f"|Z| = {spline(r2_bis):.4f} ohm")

# ============================================================
# 6. MÉTODO DE NEWTON-RAPHSON
# ============================================================

def newton_raphson(func, dfunc, x0, tol=1e-10, max_iter=50):
    x = x0

    for k in range(max_iter):
        fx = func(x)
        dfx = dfunc(x)

        if abs(dfx) < 1e-12:
            raise ValueError("La derivada es demasiado pequeña. Newton puede fallar.")

        x_new = x - fx / dfx

        if abs(x_new - x) < tol:
            return x_new, k + 1

        x = x_new

    return x, max_iter

# Aproximaciones iniciales cercanas a las raíces
r1_newton, it1_newton = newton_raphson(g, spline_d1, 110)
r2_newton, it2_newton = newton_raphson(g, spline_d1, 2200)

print("\nMétodo de Newton-Raphson:")
print("--------------------------------")
print(f"Raíz baja = {r1_newton:.6f} Hz")
print(f"Iteraciones = {it1_newton}")
print(f"|Z| = {spline(r1_newton):.6f} ohm")

print("--------------------------------")
print(f"Raíz alta = {r2_newton:.6f} Hz")
print(f"Iteraciones = {it2_newton}")
print(f"|Z| = {spline(r2_newton):.6f} ohm")

# ============================================================
# 7. SENSIBILIDAD df/dZ EN LA RAÍZ ALTA
# ============================================================

raiz_alta = r2_newton

pendiente = spline_d1(raiz_alta)

sensibilidad = 1 / pendiente

print("\nSensibilidad cerca de la raíz alta:")
print("--------------------------------")
print(f"Raíz alta = {raiz_alta:.4f} Hz")
print(f"d|Z|/df = {pendiente:.4f} ohm/Hz")
print(f"df/dZ ≈ {sensibilidad:.4f} Hz/ohm")

print("\nInterpretación:")
print(f"Un cambio de 1 ohm en la medición de impedancia podría mover la raíz aproximadamente {sensibilidad:.2f} Hz.")