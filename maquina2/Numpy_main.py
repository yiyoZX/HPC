import numpy as np
import time
import json
import os
import sys
from red import RedNeuronal
# Ejemplo con datos sintéticos más complejos
def ejemplo_clasificacion(n):
    print("\n=== Ejemplo: Clasificación con datos sintéticos ===")
    
    # Generar datos sintéticos
    np.random.seed(42)
    n_muestras = n
    
    # Clase 0: círculo interior
    r1 = np.random.uniform(0, 1, n_muestras//2)
    theta1 = np.random.uniform(0, 2*np.pi, n_muestras//2)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    
    # Clase 1: anillo exterior
    r2 = np.random.uniform(1.5, 2.5, n_muestras//2)
    theta2 = np.random.uniform(0, 2*np.pi, n_muestras//2)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    
    # Combinar datos
    X = np.column_stack([
        np.concatenate([x1, x2]),
        np.concatenate([y1, y2])
    ]).T
    
    y = np.concatenate([
        np.zeros(n_muestras//2),
        np.ones(n_muestras//2)
    ]).reshape(1, -1)
    
    # Normalizar datos
    X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
    
    print(f"Datos generados: {X.shape[1]} muestras, {X.shape[0]} características")
    
    # Crear red neuronal más compleja
    red = RedNeuronal([2, 8, 4, 1])
    
    start_time = time.time()
    print("Entrenando la red neuronal...")
    red.entrenar(X, y, epocas=1500, tasa_aprendizaje=0.1)
    end_time = time.time()
    
    print(f"Tiempo de entrenamiento: {end_time - start_time:.4f} segundos")
    
    # Evaluar
    precision = red.precision(X, y)
    print(f"Precisión final: {precision * 100:.1f}%")
    
    ruta_salida = "./datos"
    os.makedirs(ruta_salida, exist_ok=True)

    
    resultado = {
        "precision": float(red.precision(X, y)),
        "tiempo": end_time - start_time
    }

    with open(os.path.join(ruta_salida, "resultados_Numpy.json"), "w") as f:
        json.dump(resultado, f)
    
    return red

if __name__ == "__main__":
    # Ejecutar ejemplos
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    red_clasificacion = ejemplo_clasificacion()
    
    print("\n=== Red neuronal completada ===")
    print("La red neuronal incluye:")
    print("- Propagación hacia adelante y hacia atrás")
    print("- Funciones de activación (ReLU y Sigmoid)")
    print("- Entrenamiento con descenso de gradiente")
    print("- Ejemplos prácticos (XOR y clasificación)")