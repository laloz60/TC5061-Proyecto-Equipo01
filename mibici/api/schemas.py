from pydantic import BaseModel, Field, field_validator


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
    """Perfil horario de una estación para P3 (24 valores, de 0 a 23 h).

    El modelo se entrenó con perfiles normalizados por fila (proporción de
    viajes en cada hora, suma 1; ver `_perfil_ventana` en
    `mibici/modeling/train.py`). Para repetir ese preprocesamiento al servir,
    el validador normaliza la entrada: se aceptan conteos de viajes o
    proporciones y ambos producen el mismo clúster.
    """

    perfil_horario: list[float] = Field(
        ...,
        min_length=24,
        max_length=24,
        description=(
            "Viajes por hora (conteos o proporciones), de 0 a 23 h. "
            "Valores no negativos con suma mayor que cero; se normalizan a proporciones."
        ),
        examples=[[0, 0, 0, 0, 0, 1, 4, 12, 20, 10, 6, 5, 6, 6, 5, 6, 9, 15, 18, 9, 5, 3, 1, 0]],
    )

    @field_validator("perfil_horario")
    @classmethod
    def normalizar_perfil(cls, perfil: list[float]) -> list[float]:
        if any(valor < 0 for valor in perfil):
            raise ValueError("el perfil horario no admite valores negativos")
        total = sum(perfil)
        if total <= 0:
            raise ValueError("el perfil horario debe tener al menos un viaje (suma mayor que 0)")
        return [valor / total for valor in perfil]


class P1Response(BaseModel):
    demanda_predicha_viajes: float


class P2Response(BaseModel):
    riesgo_desabasto_o_saturacion: int
    probabilidad_riesgo: float


class P3Response(BaseModel):
    cluster_asignado: int
