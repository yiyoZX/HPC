# Dockerfile para el generador de gráficos
FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements si existe, o instalar dependencias directamente
RUN pip install matplotlib flask requests numpy

# Copiar archivos de la aplicación
COPY main_generador.py /app/

# Crear directorio de datos
RUN mkdir -p /datos

# Exponer puerto
EXPOSE 5002

# Comando por defecto
CMD ["python", "main_generador.py"]