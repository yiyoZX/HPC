import subprocess
import json
import os
import time

# Leer valores por input o usar por defecto
valores_input = input("🔢 Ingresa cantidades de muestras separadas por coma (ej: 1000,5000,10000):\n")
if valores_input.strip():
    valores = [int(x) for x in valores_input.split(",")]
else:
    valores = [1000, 5000, 10000]

for n in valores:
    print(f"\n Ejecutando con {n} muestras...")

    # Ejecutar contenedor de JAX
    subprocess.run(["docker", "run", "--rm", "-v", "/datos:/datos", "jax-red", str(n)])

    # Esperar archivo
    jax_resultado = f"/datos/resultados_jax_{n}.json"
    while not os.path.exists(jax_resultado):
        time.sleep(1)

    # Ejecutar contenedor de NumPy
    subprocess.run(["docker", "run", "--rm", "-v", "/datos:/datos", "numpy-red", str(n)])
    numpy_resultado = f"/datos/resultados_numpy_{n}.json"
    while not os.path.exists(numpy_resultado):
        time.sleep(1)

    # Leer y comparar
    with open(jax_resultado) as f:
        jax = json.load(f)
    with open(numpy_resultado) as f:
        numpy = json.load(f)

    print(f"\n Comparación con {n} muestras:")
    print(f"JAX   → Precisión: {jax['precision']:.4f} | Tiempo: {jax['tiempo']:.2f}s")
    print(f"NumPy → Precisión: {numpy['precision']:.4f} | Tiempo: {numpy['tiempo']:.2f}s")
