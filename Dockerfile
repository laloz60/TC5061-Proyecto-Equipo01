# Contenedor de la API REST del proyecto MiBici (Actividad 8 - Equipo 01).
# Estándar OCI: funciona igual con Docker y con Podman.
FROM python:3.12-slim

WORKDIR /app

# Capa de dependencias primero para aprovechar la caché de Docker
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código del proyecto
COPY mibici/ ./mibici/
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
RUN pip install --no-cache-dir --no-deps .

# Servidor MLflow en la máquina anfitriona (issue #124)
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000

# Reemplazar api.main:app por el módulo que suba Jesús (issue #127)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
