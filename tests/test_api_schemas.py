"""Pruebas del esquema de entrada de P3 (issue #194).

El modelo de P3 se entrenó con perfiles horarios normalizados (cada fila suma 1).
La API debe repetir ese preprocesamiento: conteos y proporciones del mismo perfil
tienen que llegar idénticos al modelo.
"""

from pydantic import ValidationError
import pytest

from mibici.api.schemas import ClusterInput

OFICINA = [0, 0, 0, 0, 0, 1, 4, 12, 20, 10, 6, 5, 6, 6, 5, 6, 9, 15, 18, 9, 5, 3, 1, 0]


def test_conteos_se_normalizan_a_proporciones():
    perfil = ClusterInput(perfil_horario=OFICINA).perfil_horario
    assert sum(perfil) == pytest.approx(1.0)
    assert perfil[8] == pytest.approx(20 / sum(OFICINA))


def test_conteos_y_proporciones_llegan_iguales_al_modelo():
    total = sum(OFICINA)
    proporciones = [x / total for x in OFICINA]
    desde_conteos = ClusterInput(perfil_horario=OFICINA).perfil_horario
    desde_proporciones = ClusterInput(perfil_horario=proporciones).perfil_horario
    assert desde_conteos == pytest.approx(desde_proporciones)


def test_perfil_ya_normalizado_no_cambia():
    plano = [1 / 24] * 24
    assert ClusterInput(perfil_horario=plano).perfil_horario == pytest.approx(plano)


@pytest.mark.parametrize(
    "perfil",
    [
        [0.0] * 24,  # sin viajes: no hay perfil que normalizar
        [-1.0] + [1.0] * 23,  # valores negativos
        [1.0] * 23,  # longitud distinta de 24
    ],
    ids=["todo-ceros", "negativo", "23-horas"],
)
def test_perfiles_invalidos_se_rechazan(perfil):
    with pytest.raises(ValidationError):
        ClusterInput(perfil_horario=perfil)
