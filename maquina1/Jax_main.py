import time
import json
import os
import sys
import jax.numpy as jnp
import jax.random as rnd
from RedJax import RedNeuronalOptimizada


def generar_datos_clasificacion(key, n_muestras):
    """Genera datos de clasificación fuera de la función JIT"""
    key1, key2, key3, key4 = rnd.split(key, 4)
    
    n_half = n_muestras // 2
    
    # Clase 0: círculo interior
    r1 = rnd.uniform(key1, shape=(n_half,), minval=0.0, maxval=1.0)
    theta1 = rnd.uniform(key2, shape=(n_half,), minval=0.0, maxval=2 * jnp.pi)
    x1 = r1 * jnp.cos(theta1)
    y1 = r1 * jnp.sin(theta1)
    
    # Clase 1: anillo exterior
    r2 = rnd.uniform(key3, shape=(n_half,), minval=1.5, maxval=2.5)
    theta2 = rnd.uniform(key4, shape=(n_half,), minval=0.0, maxval=2 * jnp.pi)
    x2 = r2 * jnp.cos(theta2)
    y2 = r2 * jnp.sin(theta2)
    
    # Combinar
    X = jnp.stack([
        jnp.concatenate([x1, x2]),
        jnp.concatenate([y1, y2])
    ])
    
    y = jnp.concatenate([
        jnp.zeros(n_half),
        jnp.ones(n_half)
    ]).reshape(1, -1)
    
    # Normalizar
    X = (X - jnp.mean(X, axis=1, keepdims=True)) / jnp.std(X, axis=1, keepdims=True)
    
    return X, y

def ejemplo_clasificacion_optimizado():
    print("\n=== Ejemplo Clasificación Optimizado ===")
    
    # Generar datos usando JAX
    clave = rnd.PRNGKey(42)
    n_muestras = 100000  # Más muestras para probar la optimización
    
    # Generar datos fuera de la función JIT
    X, y = generar_datos_clasificacion(clave, n_muestras)
    print(f"Datos generados: {X.shape[1]} muestras")
    
    # Red más profunda
    red = RedNeuronalOptimizada([2, 16, 8, 1], clave_seed=42)
    
    # Medir tiempo
    start_time = time.time()
    costos = red.entrenar(X, y, epocas=1500, tasa_aprendizaje=0.1)
    end_time = time.time()
    
    print(f"Tiempo de entrenamiento: {end_time - start_time:.4f} segundos")
    
    precision = red.precision(X, y)
    print(f"Precisión final: {float(precision) * 100:.1f}%")
    
    ruta_salida = "./datos"
    os.makedirs(ruta_salida, exist_ok=True)

    
    resultado = {
        "precision": float(red.precision(X, y)),
        "tiempo": end_time - start_time
    }

    with open(os.path.join(ruta_salida, "resultados_jax.json"), "w") as f:
        json.dump(resultado, f)

    print("Resultado JAX guardado.")
    
    return red


if __name__ == "__main__":
    # Ejecutar ejemplos
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    red_clasificacion = ejemplo_clasificacion_optimizado(n)
    red_clasificacion = ejemplo_clasificacion_optimizado()
    
    print("\n=== Red neuronal optimizada completada ===")
