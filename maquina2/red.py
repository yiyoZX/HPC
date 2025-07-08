import numpy as np

class RedNeuronal:
    def __init__(self, capas):
        """
        Inicializa la red neuronal
        capas: lista con el número de neuronas en cada capa
        Ejemplo: [2, 4, 3, 1] = 2 entradas, 2 capas ocultas (4 y 3 neuronas), 1 salida
        """
        self.num_capas = len(capas)
        self.capas = capas
        
        # Inicializar pesos y sesgos aleatoriamente
        self.pesos = []
        self.sesgos = []
        
        for i in range(1, len(capas)):
            # Pesos con inicialización Xavier/Glorot
            w = np.random.randn(capas[i], capas[i-1]) * np.sqrt(2.0 / capas[i-1])
            b = np.zeros((capas[i], 1))
            
            self.pesos.append(w)
            self.sesgos.append(b)
    
    def sigmoid(self, z):
        """Función de activación sigmoid"""
        # Evitar overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivada(self, z):
        """Derivada de la función sigmoid"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def relu(self, z):
        """Función de activación ReLU"""
        return np.maximum(0, z)
    
    def relu_derivada(self, z):
        """Derivada de ReLU"""
        return (z > 0).astype(float)
    
    def propagacion_adelante(self, x):
        """
        Propagación hacia adelante
        x: datos de entrada (características x muestras)
        """
        activacion = x
        activaciones = [x]  # almacenar activaciones
        zs = []  # almacenar valores z
        
        for i in range(len(self.pesos)):
            z = np.dot(self.pesos[i], activacion) + self.sesgos[i]
            zs.append(z)
            
            # Usar ReLU para capas ocultas y sigmoid para la última
            if i == len(self.pesos) - 1:
                activacion = self.sigmoid(z)
            else:
                activacion = self.relu(z)
            
            activaciones.append(activacion)
        
        return activaciones, zs
    
    def costo(self, y_pred, y_true):
        """Función de costo (error cuadrático medio)"""
        m = y_true.shape[1]
        return np.sum((y_pred - y_true) ** 2) / (2 * m)
    
    def propagacion_atras(self, activaciones, zs, y_true):
        """
        Propagación hacia atrás (backpropagation)
        """
        m = y_true.shape[1]  # número de muestras
        
        # Gradientes
        grad_pesos = [np.zeros(w.shape) for w in self.pesos]
        grad_sesgos = [np.zeros(b.shape) for b in self.sesgos]
        
        # Error en la capa de salida
        delta = (activaciones[-1] - y_true) * self.sigmoid_derivada(zs[-1])
        
        # Gradientes para la última capa
        grad_pesos[-1] = np.dot(delta, activaciones[-2].T) / m
        grad_sesgos[-1] = np.sum(delta, axis=1, keepdims=True) / m
        
        # Propagar error hacia atrás
        for l in range(2, self.num_capas):
            z = zs[-l]
            delta = np.dot(self.pesos[-l+1].T, delta) * self.relu_derivada(z)
            
            grad_pesos[-l] = np.dot(delta, activaciones[-l-1].T) / m
            grad_sesgos[-l] = np.sum(delta, axis=1, keepdims=True) / m
        
        return grad_pesos, grad_sesgos
    
    def entrenar(self, X, y, epocas=1000, tasa_aprendizaje=0.01, mostrar_progreso=True):
        """
        Entrenar la red neuronal
        X: datos de entrada (características x muestras)
        y: etiquetas verdaderas (salidas x muestras)
        """
        costos = []
        
        for epoca in range(epocas):
            # Propagación hacia adelante
            activaciones, zs = self.propagacion_adelante(X)
            
            # Calcular costo
            costo_actual = self.costo(activaciones[-1], y)
            costos.append(costo_actual)
            
            # Propagación hacia atrás
            grad_pesos, grad_sesgos = self.propagacion_atras(activaciones, zs, y)
            
            # Actualizar pesos y sesgos
            for i in range(len(self.pesos)):
                self.pesos[i] -= tasa_aprendizaje * grad_pesos[i]
                self.sesgos[i] -= tasa_aprendizaje * grad_sesgos[i]
            
            # Mostrar progreso
            if mostrar_progreso and epoca % 100 == 0:
                print(f'Época {epoca}, Costo: {costo_actual:.6f}')
        
        return costos
    
    def predecir(self, X):
        """Hacer predicciones"""
        activaciones, _ = self.propagacion_adelante(X)
        return activaciones[-1]
    
    def precision(self, X, y, umbral=0.5):
        """Calcular precisión para clasificación binaria"""
        predicciones = self.predecir(X)
        predicciones_binarias = (predicciones > umbral).astype(int)
        return np.mean(predicciones_binarias == y)