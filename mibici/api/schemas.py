from pydantic import BaseModel, Field


class ModelFeaturesInput(BaseModel):
    """Las 15 features exactas del input schema de los modelos P1/P2.

    Verificado contra el Model Registry: los modelos no reciben
    estacion_id, y sí incluyen los lags y medias históricas.
    """

    hora_del_dia: int = Field(ge=0, le=23)
    dia_semana: int = Field(ge=0, le=6)
    mes: int = Field(ge=1, le=12)
    es_fin_de_semana: int = Field(ge=0, le=1)
    salidas_lag_1h: float
    salidas_lag_24h: float
    salidas_lag_168h: float
    salidas_media_7d: float
    salidas_prom_4sem: float
    neto_lag_1h: float
    neto_lag_24h: float
    neto_media_7d: float
    capacidad: float
    latitud: float
    longitud: float


class ClusterInput(BaseModel):
    perfil_horario: list[float] = Field(..., min_length=24, max_length=24)


class P1Response(BaseModel):
    demanda_predicha_viajes: float


class P2Response(BaseModel):
    riesgo_desabasto_o_saturacion: int
    probabilidad_riesgo: float


class P3Response(BaseModel):
    cluster_asignado: int
