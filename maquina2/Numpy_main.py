import requests
import time
import json
import os
import numpy as np
from red import RedNeuronal

ORQUESTADOR_URL = "http://orquestador:5000"

def generar_datos_clasificacion(n_muestras, seed=42):
    np.random.seed(seed)
    n_half = n_muestras // 2
    r1 = np.random.uniform(0.0, 1.0, n_half)
    theta1 = np.random.uniform(0.0, 2 * np.pi, n_half)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    r2 = np.random.uniform(1.5, 2.5, n_half)
    theta2 = np.random.uniform(0.0, 2 * np.pi, n_half)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    X = np.vstack([
        np.concatenate([x1, x2]),
        np.concatenate([y1, y2])
    ])
    y = np.concatenate([
        np.zeros(n_half),
        np.ones(n_half)
    ]).reshape(1, -1)
    X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
    return X, y

def ejecutar_tarea(n):
    np.random.seed(42)  # Inicializa la semilla aquí
    X, y = generar_datos_clasificacion(n)
    red = RedNeuronal([2, 16, 8, 1])  # Sin argumento seed
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
    with open(os.path.join(ruta_salida, f"resultados_numpy_{n}.json"), "w") as f:
        json.dump(resultado, f)
    print(f"Resultado NumPy guardado para n={n}.")
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
                "quien": "numpy",
                "n": n,
                "resultado": resultado
            })
            print("Respuesta del orquestador:", resp.status_code, resp.text)
        except Exception as e:
            print("Error:", e)
            time.sleep(2)