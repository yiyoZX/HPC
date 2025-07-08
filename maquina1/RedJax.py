import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import jit, grad, vmap
from functools import partial

# Compatibilidad con diferentes versiones de JAX
try:
    from jax import tree
    tree_map = tree.map
except (ImportError, AttributeError):
    # Para versiones más antiguas de JAX
    try:
        tree_map = jax.tree.map
    except AttributeError:
        tree_map = jax.tree_util.tree_map

class RedNeuronalOptimizada:
    def __init__(self, capas, clave_seed=0):
        """
        Inicializa la red neuronal optimizada con JAX
        capas: lista con el número de neuronas en cada capa
        """
        self.num_capas = len(capas)
        self.capas = capas
        
        # Inicializar parámetros
        self.clave = rnd.PRNGKey(clave_seed)
        self.params = self._inicializar_parametros()
        
        # Compilar funciones JIT para máximo rendimiento
        self._compilar_funciones()
    
    def _inicializar_parametros(self):
        """Inicializa parámetros usando una estructura más eficiente"""
        params = {'pesos': [], 'sesgos': []}
        
        for i in range(1, len(self.capas)):
            self.clave, subclave = rnd.split(self.clave)
            
            # Inicialización He para ReLU
            fan_in = self.capas[i-1]
            w = rnd.normal(subclave, (self.capas[i], fan_in)) * jnp.sqrt(2.0 / fan_in)
            b = jnp.zeros((self.capas[i], 1))
            
            params['pesos'].append(w)
            params['sesgos'].append(b)
        
        return params
    
    def _compilar_funciones(self):
        """Compila las funciones principales con JIT para máximo rendimiento"""
        
        @jit
        def propagacion_adelante_jit(params, x):
            """Propagación hacia adelante compilada con JIT"""
            activacion = x
            
            for i, (w, b) in enumerate(zip(params['pesos'], params['sesgos'])):
                z = jnp.dot(w, activacion) + b
                
                # ReLU para capas ocultas, sigmoid para salida
                if i == len(params['pesos']) - 1:
                    activacion = self._sigmoid(z)
                else:
                    activacion = jnp.maximum(0, z)
            
            return activacion
        
        @jit
        def costo_jit(params, x, y):
            """Función de costo compilada"""
            pred = propagacion_adelante_jit(params, x)
            return jnp.mean((pred - y) ** 2)
        
        # Compilar gradientes automáticamente
        @jit
        def gradientes_jit(params, x, y):
            """Gradientes compilados automáticamente"""
            return grad(costo_jit)(params, x, y)
        
        @jit
        def actualizar_parametros_jit(params, grads, lr):
            """Actualización de parámetros compilada"""
            return tree_map(lambda p, g: p - lr * g, params, grads)
        
        # Vectorización para múltiples muestras usando vmap
        @jit 
        def predecir_muestra_jit(params, x):
            """Predicción para una sola muestra"""
            return propagacion_adelante_jit(params, x.reshape(-1, 1))
        
        # Predicción vectorizada para múltiples muestras
        @jit
        def predecir_lote_jit(params, X):
            """Predicción vectorizada para múltiples muestras usando vmap"""
            # vmap sobre la segunda dimensión (muestras)
            predecir_vectorizada = vmap(
                lambda x: propagacion_adelante_jit(params, x.reshape(-1, 1)).flatten(),
                in_axes=1, out_axes=0
            )
            return predecir_vectorizada(X).reshape(1, -1)
        
        # Asignar funciones compiladas
        self.propagacion_adelante_jit = propagacion_adelante_jit
        self.costo_jit = costo_jit
        self.gradientes_jit = gradientes_jit
        self.actualizar_parametros_jit = actualizar_parametros_jit
        self.predecir_muestra_jit = predecir_muestra_jit
        self.predecir_lote_jit = predecir_lote_jit
    
    @staticmethod
    def _sigmoid(z):
        """Función sigmoid optimizada"""
        return jax.nn.sigmoid(z)  # Usa la implementación optimizada de JAX
    
    def entrenar(self, X, y, epocas=1000, tasa_aprendizaje=0.01, mostrar_progreso=True):
        """
        Entrenamiento optimizado con JAX
        """
        costos = []
        params = self.params
        
        # Compilar el paso de entrenamiento completo
        @jit
        def paso_entrenamiento(params, x, y, lr):
            grads = self.gradientes_jit(params, x, y)
            params_nuevos = self.actualizar_parametros_jit(params, grads, lr)
            costo = self.costo_jit(params, x, y)
            return params_nuevos, costo
        
        print("Compilando funciones JIT... (primera época será más lenta)")
        
        for epoca in range(epocas):
            # Un solo paso de entrenamiento compilado
            params, costo_actual = paso_entrenamiento(params, X, y, tasa_aprendizaje)
            costos.append(float(costo_actual))
            
            if mostrar_progreso and epoca % 100 == 0:
                print(f'Época {epoca}, Costo: {costo_actual:.6f}')
        
        self.params = params
        return costos
    
    def predecir(self, X):
        """Predicción optimizada y vectorizada"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Usar predicción vectorizada compilada para múltiples muestras
        if X.shape[1] > 1:
            return self.predecir_lote_jit(self.params, X)
        else:
            pred = self.propagacion_adelante_jit(self.params, X)
            return pred
    
    def precision(self, X, y, umbral=0.5):
        """Precisión optimizada con predicción vectorizada"""
        predicciones = self.predecir(X)
        if predicciones.ndim == 1:
            predicciones = predicciones.reshape(1, -1)
        elif predicciones.shape[0] != 1:
            predicciones = predicciones.T
            
        predicciones_binarias = (predicciones > umbral).astype(int)
        return jnp.mean(predicciones_binarias == y)
