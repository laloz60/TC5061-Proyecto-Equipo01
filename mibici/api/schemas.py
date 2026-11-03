from pydantic import BaseModel, Field
from typing import List

class ModelFeaturesInput(BaseModel):
    estacion_id: int
    hora_del_dia: int
    dia_semana: int
    es_fin_de_semana: int
    mes: int
    capacidad: float
    latitud: float
    longitud: float
    salidas_lag_1h: float = 0.0
    neto_lag_1h: float = 0.0

class ClusterInput(BaseModel):
    perfil_horario: List[float] = Field(..., min_length=24, max_length=24)

class P1Response(BaseModel):
    demanda_predicha_viajes: float

class P2Response(BaseModel):
    riesgo_desabasto_o_saturacion: int
    probabilidad_riesgo: float

class P3Response(BaseModel):
    cluster_asignado: int