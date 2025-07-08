import requests
import time
import json
import os
import jax.numpy as jnp
import jax.random as rnd
from RedJax import RedNeuronalOptimizada

ORQUESTADOR_URL = "http://orquestador:5000"

def generar_datos_clasificacion(key, n_muestras):
    key1, key2, key3, key4 = rnd.split(key, 4)
    n_half = n_muestras // 2
    r1 = rnd.uniform(key1, shape=(n_half,), minval=0.0, maxval=1.0)
    theta1 = rnd.uniform(key2, shape=(n_half,), minval=0.0, maxval=2 * jnp.pi)
    x1 = r1 * jnp.cos(theta1)
    y1 = r1 * jnp.sin(theta1)
    r2 = rnd.uniform(key3, shape=(n_half,), minval=1.5, maxval=2.5)
    theta2 = rnd.uniform(key4, shape=(n_half,), minval=0.0, maxval=2 * jnp.pi)
    x2 = r2 * jnp.cos(theta2)
    y2 = r2 * jnp.sin(theta2)
    X = jnp.stack([
        jnp.concatenate([x1, x2]),
        jnp.concatenate([y1, y2])
    ])
    y = jnp.concatenate([
        jnp.zeros(n_half),
        jnp.ones(n_half)
    ]).reshape(1, -1)
    X = (X - jnp.mean(X, axis=1, keepdims=True)) / jnp.std(X, axis=1, keepdims=True)
    return X, y

def ejecutar_tarea(n):
    clave = rnd.PRNGKey(42)
    X, y = generar_datos_clasificacion(clave, n)
    red = RedNeuronalOptimizada([2, 16, 8, 1], clave_seed=42)
    start_time = time.time()
    red.entrenar(X, y, epocas=1500, tasa_aprendizaje=0.1)
    end_time = time.time()
    precision = float(red.precision(X, y))
    tiempo = end_time - start_time
    resultado = {
        "precision": precision,
        "tiempo": tiempo
    }
    ruta_salida = "/datos"
    os.makedirs(ruta_salida, exist_ok=True)
    with open(os.path.join(ruta_salida, f"resultados_jax_{n}.json"), "w") as f:
        json.dump(resultado, f)
    print(f"Resultado JAX guardado para n={n}.")
    return resultado

if __name__ == "__main__":
    # Esperar a que el orquestador esté disponible
    while True:
        try:
            resp = requests.get(f"{ORQUESTADOR_URL}/tarea", timeout=5)
            if resp.status_code == 200:
                break
        except Exception as e:
            print("Esperando al orquestador...", e)
            time.sleep(2)

    resp = requests.get(f"{ORQUESTADOR_URL}/tarea")
    cantidades = resp.json().get("cantidades", [])
    for n in cantidades:
        try:
            print(f"Procesando tarea con n={n}")
            resultado = ejecutar_tarea(n)
            resp = requests.post(f"{ORQUESTADOR_URL}/resultado", json={
                "quien": "jax",
                "n": n,
                "resultado": resultado
            })
            print("Respuesta del orquestador:", resp.status_code, resp.text)
        except Exception as e:
            print("Error:", e)
            time.sleep(2)