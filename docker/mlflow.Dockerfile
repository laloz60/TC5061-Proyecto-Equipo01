# Contenedor del servidor de tracking MLflow (Entregable 7 - issue #156).
# Estándar OCI: funciona igual con Docker y con Podman.
#
# La API no guarda los modelos en su imagen: los descarga del Model Registry
# que corre en este contenedor, por eso el sistema necesita los dos servicios.
FROM python:3.12-slim

WORKDIR /app

# Solo el servidor de tracking. La versión va fijada igual que en
# requirements-api.txt para que API y servidor compartan la misma versión de
# MLflow (el protocolo REST y el Model Registry deben coincidir).
RUN pip install --no-cache-dir mlflow==3.17.2

# Ambos artefactos se montan desde el host en docker-compose.yml:
#   /app/mlflow.db -> backend store (Model Registry, versionado)
#   /app/mlruns    -> artefactos de los modelos (1.3 GB, no se copian a la imagen)
EXPOSE 5000

# curl no esta en python:3.12-slim, el chequeo usa urllib contra /health.
# start-period de 30s: arranca 4 workers y levanta el esquema de SQLite.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)" || exit 1

# --host 0.0.0.0: si no, el servidor solo escucha en 127.0.0.1 y la API del
# otro contenedor no lo alcanza (ver tabla de errores del README).
# --workers 2: con los 4 por defecto el servidor se lleva ~1.9 GB de RAM y
# llega al 92% del limite de 2g del compose; la API solo consulta el registry
# al arrancar, asi que 2 workers bastan y dejan el consumo en ~1 GB.
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////app/mlflow.db", \
     "--artifacts-destination", "/app/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--workers", "2"]
