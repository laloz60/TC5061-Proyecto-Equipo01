# Usa una imagen oficial de Python ligera
FROM python:3.12.9-slim

# Evita que Python genere archivos .pyc y fuerza salida en consola sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configura el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala dependencias del sistema necesarias para compilar librerías (ej. numpy, scipy)
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copia los archivos de requerimientos e instálalos
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia todo el código del proyecto al contenedor
COPY . /app/

# Instala el paquete local en modo edición para que Python reconozca el módulo `mibici`
RUN pip install -e .

# Expone el puerto que usará FastAPI
EXPOSE 8000

# Comando para iniciar el servidor web con Uvicorn
CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]