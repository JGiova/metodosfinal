import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def validar_float(prompt):
    """Valida y retorna un número float introducido por el usuario."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Por favor, introduce un número válido.")

def ejercicio1():
    print("\nEjercicio 1: Maximización de rendimientos con restricciones")

    # Entrada de datos
    r1 = validar_float("Introduce el rendimiento del activo 1 (0-1): ")
    r2 = validar_float("Introduce el rendimiento del activo 2 (0-1): ")
    risk1 = validar_float("Introduce el riesgo del activo 1 (0-1): ")
    risk2 = validar_float("Introduce el riesgo del activo 2 (0-1): ")
    riesgo_max = validar_float("Introduce el riesgo máximo permitido (0-1): ")

    # Función objetivo
    def objective(x):
        return -(r1 * x[0] + r2 * x[1])  # Negativo porque queremos maximizar

    # Restricción de presupuesto
    def constraint1(x):
        return x[0] + x[1] - 1

    # Restricción de riesgo
    def constraint2(x):
        return riesgo_max - (risk1 * x[0]**2 + risk2 * x[1]**2)

    x0 = [0.5, 0.5]  # Punto inicial
    cons = ({'type': 'eq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2})

    # Optimización
    sol = minimize(objective, x0, method='SLSQP', constraints=cons)

    print("\nSolución óptima:")
    print(f"Inversión en activo 1 = {sol.x[0]:.4f}")
    print(f"Inversión en activo 2 = {sol.x[1]:.4f}")
    print(f"Rendimiento máximo = {-sol.fun:.4f}")

def ejercicio2():
    print("\nEjercicio 2: Minimización de costos con restricción")

    # Entrada de datos
    c1 = validar_float("Introduce el coeficiente de costo para el producto A: ")
    c2 = validar_float("Introduce el coeficiente de costo para el producto B: ")
    c3 = validar_float("Introduce el coeficiente de costo para el producto C: ")
    total_produccion = validar_float("Introduce la producción total requerida: ")

    # Función objetivo
    def objective(x):
        return c1 * x[0]**2 + c2 * x[1]**2 + c3 * x[2]**2

    # Restricción de producción total
    def constraint(x):
        return x[0] + x[1] + x[2] - total_produccion

    x0 = [total_produccion/3, total_produccion/3, total_produccion/3]  # Punto inicial
    cons = {'type': 'eq', 'fun': constraint}

    # Optimización
    sol = minimize(objective, x0, method='SLSQP', constraints=cons)

    print("\nSolución óptima:")
    print(f"Producción de A = {sol.x[0]:.2f}")
    print(f"Producción de B = {sol.x[1]:.2f}")
    print(f"Producción de C = {sol.x[2]:.2f}")
    print(f"Costo mínimo = {sol.fun:.2f}")

    # Gráfica
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, total_produccion, total_produccion/10)
    X, Y = np.meshgrid(x, y)
    Z = total_produccion - X - Y
    C = c1 * X**2 + c2 * Y**2 + c3 * Z**2

    ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(C/C.max()))
    ax.scatter(sol.x[0], sol.x[1], sol.x[2], color='r', s=100)
    ax.set_xlabel('Producto A')
    ax.set_ylabel('Producto B')
    ax.set_zlabel('Producto C')
    ax.set_title('Distribución óptima de producción')

    plt.show()

def ejercicio3():
    print("\nEjercicio 3: Descenso del gradiente")

    # Entrada de datos
    a = validar_float("Introduce el coeficiente a: ")
    b = validar_float("Introduce el coeficiente b: ")
    c = validar_float("Introduce el coeficiente c: ")
    d = validar_float("Introduce el coeficiente d: ")
    e = validar_float("Introduce el coeficiente e: ")
    learning_rate = validar_float("Introduce la tasa de aprendizaje (0-1): ")
    num_iterations = int(validar_float("Introduce el número de iteraciones: "))

    # Función objetivo
    def f(x, y, z):
        return a*x**2 + b*y**2 + c*z**2 + d*x*y + e*z

    # Gradiente de la función
    def gradient(x, y, z):
        dx = 2*a*x + d*y
        dy = 2*b*y + d*x
        dz = 2*c*z + e
        return np.array([dx, dy, dz])

    # Algoritmo de descenso del gradiente
    def gradient_descent(start, learning_rate, num_iterations):
        path = [start]
        for _ in range(num_iterations):
            grad = gradient(*path[-1])
            new_point = path[-1] - learning_rate * grad
            path.append(new_point)
        return np.array(path)

    start = np.array([1, 1, 1])
    path = gradient_descent(start, learning_rate, num_iterations)

    # Gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_iterations + 1), [f(*p) for p in path], '-o')
    plt.xlabel('Iteración')
    plt.ylabel('f(x, y, z)')
    plt.title('Evolución de f(x, y, z) durante el descenso del gradiente')
    plt.grid(True)
    plt.show()

    print("\nResultados:")
    print("Punto final:", path[-1])
    print("Valor final de f(x,y,z):", f(*path[-1]))

def ejercicio4():
    print("\nEjercicio 4: Optimización con restricciones de desigualdad")

    # Entrada de datos
    a = validar_float("Introduce el coeficiente a: ")
    b = validar_float("Introduce el coeficiente b: ")
    c = validar_float("Introduce el coeficiente c: ")
    x_min = validar_float("Introduce el valor mínimo de x: ")
    x_max = validar_float("Introduce el valor máximo de x: ")

    # Función objetivo
    def objective(x):
        return a*x**2 + b*x + c

    # Restricciones
    def constraint1(x):
        return x - x_min

    def constraint2(x):
        return x_max - x

    x0 = (x_min + x_max) / 2  # Punto inicial
    cons = ({'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2})

    # Optimización
    sol = minimize(objective, x0, method='SLSQP', constraints=cons)

    print("\nSolución óptima:")
    print(f"x = {sol.x[0]:.4f}")
    print(f"Valor mínimo = {sol.fun:.4f}")

def ejercicio5():
    print("\nEjercicio 5: Puntos estacionarios")

    # Entrada de datos
    a = validar_float("Introduce el coeficiente a: ")
    b = validar_float("Introduce el coeficiente b: ")
    c = validar_float("Introduce el coeficiente c: ")
    d = validar_float("Introduce el coeficiente d: ")

    # Función cúbica
    def f(x):
        return a*x**3 + b*x**2 + c*x + d

    # Derivada de la función
    def df(x):
        return 3*a*x**2 + 2*b*x + c

    x = np.linspace(-10, 10, 1000)
    y = f(x)

    # Encontrar puntos estacionarios
    roots = np.roots([3*a, 2*b, c])
    stationary_points = roots[np.isreal(roots)].real

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.plot(stationary_points, f(stationary_points), 'ro')

    for point in stationary_points:
        if df(point-0.001) < 0 and df(point+0.001) > 0:
            plt.annotate('Mínimo', (point, f(point)), xytext=(5, 5), textcoords='offset points')
        elif df(point-0.001) > 0 and df(point+0.001) < 0:
            plt.annotate('Máximo', (point, f(point)), xytext=(5, 5), textcoords='offset points')
        else:
            plt.annotate('Punto de inflexión', (point, f(point)), xytext=(5, 5), textcoords='offset points')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Puntos estacionarios de f(x) = {a}x³ + {b}x² + {c}x + {d}')
    plt.grid(True)
    plt.show()

def menu():
    while True:
        print("\n--- Menú Principal ---")
        print("1. Maximización de rendimientos con restricciones")
        print("2. Minimización de costos con restricción")
        print("3. Descenso del gradiente")
        print("4. Optimización con restricciones de desigualdad")
        print("5. Puntos estacionarios")
        print("0. Salir")
        
        opcion = input("Seleccione un ejercicio (0-5): ")
        
        if opcion == '1':
            ejercicio1()
        elif opcion == '2':
            ejercicio2()
        elif opcion == '3':
            ejercicio3()
        elif opcion == '4':
            ejercicio4()
        elif opcion == '5':
            ejercicio5()
        elif opcion == '0':
            print("Gracias por usar el programa. ¡Hasta luego!")
            break
        else:
            print("Opción no válida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    menu()