from pydantic import BaseModel, Field, ConfigDict
from typing import List

# ===================
# ESQUEMAS DE ENTRADA
# ===================

class ModelFeaturesInput(BaseModel):
    """
    Esquema base para P1 (Regresión) y P2 (Clasificación).
    Valida las columnas obligatorias conocidas y permite dinámicamente 
    las columnas de lags mediante 'extra="allow"'.
    """
    estacion_id: int = Field(..., description="ID oficial de la estación")
    hora_del_dia: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    dia_semana: int = Field(..., ge=0, le=6, description="Lunes=0, Domingo=6")
    es_fin_de_semana: int = Field(..., ge=0, le=1, description="Booleano (0 o 1)")
    mes: int = Field(..., ge=1, le=12, description="Mes del año (1-12)")
    capacidad: float = Field(..., ge=0, description="Número de anclajes de la estación")
    latitud: float = Field(...)
    longitud: float = Field(...)

    # Permite inyectar las columnas de rezagos (lags) dinámicamente
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "estacion_id": 12,
                "hora_del_dia": 8,
                "dia_semana": 0,
                "es_fin_de_semana": 0,
                "mes": 5,
                "capacidad": 20.0,
                "latitud": 20.67,
                "longitud": -103.35,
                "salidas_lag_1h": 5.0,
                "neto_lag_1h": 2.0
            }
        }
    )

class ClusterInput(BaseModel):
    """
    Esquema para P3 (Clustering).
    Exige exactamente 24 valores flotantes (proporción de viajes por hora).
    """
    perfil_horario: List[float] = Field(
        ..., 
        min_length=24, 
        max_length=24, 
        description="Array de 24 posiciones con la proporción de viajes por hora (0.0 a 1.0)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "perfil_horario": [0.01]*5 + [0.05, 0.1, 0.15, 0.08] + [0.05]*15
            }
        }
    )

# ==================
# ESQUEMAS DE SALIDA
# ==================

class P1Response(BaseModel):
    demanda_predicha_viajes: float

class P2Response(BaseModel):
    riesgo_desabasto_o_saturacion: int
    probabilidad_riesgo: float

class P3Response(BaseModel):
    cluster_asignado: int