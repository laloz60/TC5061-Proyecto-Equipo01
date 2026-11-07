# Contenedor de la API REST del proyecto MiBici (Actividad 8 - Equipo 01).
# Estándar OCI: funciona igual con Docker y con Podman.
FROM python:3.12-slim

WORKDIR /app

# Capa de dependencias primero para aprovechar la caché de Docker
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código del proyecto
COPY mibici/ ./mibici/
# import mibici resuelve desde /app sin instalar el paquete: cambiar un .py
# solo re-ejecuta este COPY (la capa de dependencias queda cacheada)
ENV PYTHONPATH=/app

# Servidor MLflow en la máquina anfitriona (issue #124)
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000

# curl no esta en python:3.12-slim, asi que el chequeo usa urllib.
# /health responde 503 si los modelos no cargaron: el contenedor queda
# "unhealthy" en vez de decir "healthy" sin estarlo.
# start-period de 60s: la descarga de los 3 modelos @champion tarda.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]