import json
import os
from flask import Flask, request, jsonify
import requests
import threading
import time

# Leer configuración desde archivo
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"No se encontró el archivo de configuración: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

valores = config.get("cantidades", [1000, 5000, 10000])
valores_iter = iter(valores)
resultados = {}
graficos_generados = False

app = Flask(__name__)

GENERADOR_URL = "http://generador:5002"

@app.route('/tarea', methods=['GET'])
def get_tarea():
    """Endpoint para obtener las tareas a ejecutar"""
    return jsonify({"cantidades": config.get("cantidades", [1000, 5000, 10000])})

@app.route('/resultado', methods=['POST'])
def recibir_resultado():
    """Endpoint para recibir resultados de los workers"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Datos no proporcionados"}), 400
            
        quien = data.get("quien")
        n = str(data.get("n"))  # Asegurar que n es string
        resultado = data.get("resultado")
        
        if not all([quien, n, resultado]):
            return jsonify({"error": "Faltan campos requeridos"}), 400

        # Validar que 'quien' sea válido
        if quien not in ["jax", "numpy"]:
            return jsonify({"error": "Campo 'quien' debe ser 'jax' o 'numpy'"}), 400

        # Almacenar resultados
        resultados.setdefault(n, {})[quien] = resultado
        print(f"Recibido resultado de {quien} para {n}: {resultado}")

        # Guardar en archivo
        os.makedirs("/datos", exist_ok=True)
        with open("/datos/resultados_totales.json", "w") as f:
            json.dump(resultados, f, indent=2)

        # Verificar si tenemos ambos resultados para esta n
        if set(resultados[n].keys()) >= {"jax", "numpy"}:
            jax_res = resultados[n]["jax"]
            numpy_res = resultados[n]["numpy"]
            print(f"\nComparación con {n} muestras:")
            print(f"JAX   → Precisión: {jax_res['precision']:.4f} | Tiempo: {jax_res['tiempo']:.2f}s")
            print(f"NumPy → Precisión: {numpy_res['precision']:.4f} | Tiempo: {numpy_res['tiempo']:.2f}s")

        # Verificar completitud para todas las cantidades
        cantidades = set(str(v) for v in config.get("cantidades", [1000, 5000, 10000]))
        completado = cantidades.issubset(resultados.keys())
        todos_resultados = all(set(res.keys()) >= {"jax", "numpy"} for res in resultados.values())
        
        response_data = {
            "ok": True, 
            "mensaje": "Resultado almacenado",
            "completado": completado and todos_resultados,
            "total_cantidades": len(cantidades),
            "cantidades_completadas": len([k for k in resultados.keys() if set(resultados[k].keys()) >= {"jax", "numpy"}])
        }

        if completado and todos_resultados:
            print("\n¡Todas las tareas completadas!")
            response_data["mensaje"] = "Todas las tareas completadas. El generador de gráficos será notificado."

        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error en recibir_resultado: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint para obtener el estado actual de los resultados"""
    cantidades = config.get("cantidades", [1000, 5000, 10000])
    status = {}
    
    for cantidad in cantidades:
        cantidad_str = str(cantidad)
        if cantidad_str in resultados:
            status[cantidad_str] = {
                "jax": "jax" in resultados[cantidad_str],
                "numpy": "numpy" in resultados[cantidad_str],
                "completo": set(resultados[cantidad_str].keys()) >= {"jax", "numpy"}
            }
        else:
            status[cantidad_str] = {
                "jax": False,
                "numpy": False,
                "completo": False
            }
    
    total_completo = all(status[str(c)]["completo"] for c in cantidades)
    
    return jsonify({
        "status": status,
        "total_completo": total_completo,
        "graficos_generados": graficos_generados
    })

@app.route('/generar_grafico', methods=['POST'])
def solicitar_grafico():
    """Endpoint para solicitar generación de gráficos al generador"""
    try:
        # Verificar que tenemos todos los resultados
        cantidades = set(str(v) for v in config.get("cantidades", [1000, 5000, 10000]))
        
        if not cantidades.issubset(resultados.keys()):
            return jsonify({
                "error": "No se tienen todos los resultados necesarios",
                "cantidades_requeridas": list(cantidades),
                "cantidades_disponibles": list(resultados.keys())
            }), 400

        # Verificar que cada cantidad tiene resultados de JAX y NumPy
        faltantes = []
        for cantidad in cantidades:
            if cantidad not in resultados:
                faltantes.append(f"Cantidad {cantidad}: Sin resultados")
            elif not set(resultados[cantidad].keys()) >= {"jax", "numpy"}:
                disponibles = list(resultados[cantidad].keys())
                faltantes.append(f"Cantidad {cantidad}: Faltan {set(['jax', 'numpy']) - set(disponibles)}")
        
        if faltantes:
            return jsonify({
                "error": "Faltan resultados para generar gráficos",
                "detalles": faltantes
            }), 400

        # Solicitar generación de gráficos al servicio generador
        try:
            resp = requests.post(f"{GENERADOR_URL}/generar", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return jsonify({
                    "ok": True,
                    "mensaje": "Gráficos generados exitosamente",
                    "ruta_grafico": data.get("ruta")
                })
            else:
                return jsonify({
                    "error": "Error al generar gráficos en el servicio generador",
                    "detalles": resp.text
                }), 500
        except requests.exceptions.RequestException as e:
            return jsonify({
                "error": f"No se pudo conectar al generador de gráficos: {str(e)}"
            }), 500
    
    except Exception as e:
        print(f"Error al solicitar gráficos: {str(e)}")
        return jsonify({"error": f"Error al solicitar gráficos: {str(e)}"}), 500

@app.route('/grafico_listo', methods=['POST'])
def grafico_listo():
    """Endpoint para que el generador notifique que los gráficos están listos"""
    global graficos_generados
    try:
        data = request.get_json()
        if data and data.get("status") == "completado":
            graficos_generados = True
            print("✓ Gráficos generados y notificados por el generador")
            return jsonify({"ok": True, "mensaje": "Notificación recibida"})
        else:
            return jsonify({"error": "Datos de notificación inválidos"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_server():
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    run_server()