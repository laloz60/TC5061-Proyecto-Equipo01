# Info scripts/test_api.sh

from locust import HttpUser, between, task
from locust.exception import StopUser

# Payload de 15 features (P1 y P2)
PAYLOAD_P1_P2 = {
    "hora_del_dia": 8,
    "dia_semana": 5,
    "mes": 10,
    "es_fin_de_semana": 0,
    "salidas_lag_1h": 4.0,
    "salidas_lag_24h": 6.0,
    "salidas_lag_168h": 5.0,
    "salidas_media_7d": 5.5,
    "salidas_prom_4sem": 5.2,
    "neto_lag_1h": -1.0,
    "neto_lag_24h": 0.5,
    "neto_media_7d": 0.2,
    "capacidad": 30.0,
    "latitud": 20.6597,
    "longitud": -103.3496,}

# Payload P3: viajes por hora de una estación tipo oficina (la API lo normaliza)
PAYLOAD_P3 = {
    "perfil_horario": [0, 0, 0, 0, 0, 1, 4, 12, 20, 10, 6, 5, 6, 6, 5, 6, 9, 15, 18, 9, 5, 3, 1, 0],}


class MiBiciUsuario(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        with self.client.get("/health", name="GET /health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"/health respondio con status {response.status_code}")
                raise StopUser()

    @task(4)
    def predecir_p1_demanda(self):
        self.client.post("/predict/p1-demanda", json=PAYLOAD_P1_P2)

    @task(4)
    def predecir_p2_riesgo(self):
        self.client.post("/predict/p2-riesgo", json=PAYLOAD_P1_P2)

    @task(2)
    def predecir_p3_clustering(self):
        self.client.post("/predict/p3-clustering", json=PAYLOAD_P3)