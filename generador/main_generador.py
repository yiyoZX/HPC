import json
import matplotlib.pyplot as plt
import os
import re
import requests
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

ORQUESTADOR_URL = "http://orquestador:5000"
DATOS_PATH = "/datos"

def generar_grafico():
    """Genera los gráficos de comparación entre JAX y NumPy"""
    try:
        resultados = {}

        # Leer archivos de resultados de JAX
        for fname in os.listdir(DATOS_PATH):
            match = re.match(r"resultados_jax_(\d+)\.json", fname)
            if match:
                n = match.group(1)
                with open(os.path.join(DATOS_PATH, fname)) as f:
                    resultados.setdefault(n, {})["jax"] = json.load(f)

        # Leer archivos de resultados de NumPy
        for fname in os.listdir(DATOS_PATH):
            match = re.match(r"resultados_numpy_(\d+)\.json", fname)
            if match:
                n = match.group(1)
                with open(os.path.join(DATOS_PATH, fname)) as f:
                    resultados.setdefault(n, {})["numpy"] = json.load(f)

        if not resultados:
            raise Exception("No se encontraron archivos de resultados")

        n_muestras = sorted([int(k) for k in resultados.keys()])
        jax_prec = []
        jax_time = []
        numpy_prec = []
        numpy_time = []

        for n in n_muestras:
            res = resultados[str(n)]
            jax_prec.append(res.get("jax", {}).get("precision", None))
            jax_time.append(res.get("jax", {}).get("tiempo", None))
            numpy_prec.append(res.get("numpy", {}).get("precision", None))
            numpy_time.append(res.get("numpy", {}).get("tiempo", None))

        # Crear gráfico
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(n_muestras, jax_prec, marker='o', label="JAX", linewidth=2)
        plt.plot(n_muestras, numpy_prec, marker='s', label="NumPy", linewidth=2)
        plt.xlabel("Cantidad de muestras")
        plt.ylabel("Precisión")
        plt.title("Precisión vs Cantidad de muestras")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(n_muestras, jax_time, marker='o', label="JAX", linewidth=2)
        plt.plot(n_muestras, numpy_time, marker='s', label="NumPy", linewidth=2)
        plt.xlabel("Cantidad de muestras")
        plt.ylabel("Tiempo (s)")
        plt.title("Tiempo de ejecución vs Cantidad de muestras")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(DATOS_PATH, "resultados.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico generado exitosamente: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error al generar gráfico: {str(e)}")
        raise

@app.route('/generar', methods=['POST'])
def generar_grafico_endpoint():
    """Endpoint para generar gráficos"""
    try:
        output_path = generar_grafico()
        return jsonify({
            "ok": True,
            "mensaje": "Gráfico generado exitosamente",
            "ruta": output_path
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check"""
    return jsonify({"status": "healthy", "service": "generador-graficos"})

def monitorear_resultados():
    """Monitorea el orquestador para saber cuándo generar gráficos"""
    print("Iniciando monitoreo de resultados...")
    
    while True:
        try:
            # Consultar estado del orquestador
            resp = requests.get(f"{ORQUESTADOR_URL}/status", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("total_completo", False):
                    print("¡Todos los resultados están listos! Generando gráficos automáticamente...")
                    try:
                        generar_grafico()
                        print("Gráficos generados automáticamente")
                        # Notificar al orquestador que los gráficos están listos
                        requests.post(f"{ORQUESTADOR_URL}/grafico_listo", 
                                    json={"status": "completado"}, 
                                    timeout=5)
                        break  # Salir del loop una vez generados los gráficos
                    except Exception as e:
                        print(f"Error al generar gráficos automáticamente: {e}")
            else:
                print(f"Error al consultar orquestador: {resp.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión con orquestador: {e}")
        except Exception as e:
            print(f"Error inesperado en monitoreo: {e}")
        
        time.sleep(10)  # Esperar 10 segundos antes de la próxima consulta

def run_server():
    """Ejecuta el servidor Flask"""
    app.run(host="0.0.0.0", port=5002, debug=False)

if __name__ == "__main__":
    import threading
    
    # Crear directorio de datos si no existe
    os.makedirs(DATOS_PATH, exist_ok=True)
    
    # Ejecutar servidor en un hilo separado
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Ejecutar monitoreo en el hilo principal
    monitorear_resultados()