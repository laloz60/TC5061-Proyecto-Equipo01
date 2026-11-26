import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from mibici.modeling import continuous_training as ct


def _write_drift(tmp_path: Path, dataset_drift: bool, evidently_format: bool = False) -> Path:
    path = tmp_path / "drift_report.json"
    payload = {
        "dataset_drift": dataset_drift,
        "number_of_drifted_columns": 12 if dataset_drift else 4,
        "number_of_columns": 21,
        "share_of_drifted_columns": 0.571 if dataset_drift else 0.190,
    }
    if evidently_format:
        payload = {"metrics": [{"metric": "DatasetDriftMetric", "result": payload}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_leer_reporte_drift_formato_evidently(tmp_path):
    report = _write_drift(tmp_path, dataset_drift=False, evidently_format=True)
    status = ct.leer_reporte_drift(report)

    assert status.dataset_drift is False
    assert status.drifted_columns == 4
    assert status.total_columns == 21


def test_obtener_champion_lee_tags_oficiales():
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="4",
        run_id="run-123",
        tags={"metrica_seleccion": "val_roc_auc", "valor_metrica": "0.8293"},
    )

    version, metric = ct.obtener_champion_actual(
        mock_client, "mibici-equipo01-p2-riesgo", "val_roc_auc"
    )

    assert version == "4"
    assert metric == 0.8293
    mock_client.get_run.assert_not_called()


def test_sin_drift_no_reentrena(tmp_path, monkeypatch):
    report = _write_drift(tmp_path, dataset_drift=False)
    mock_client = MagicMock()
    monkeypatch.setattr(ct, "MlflowClient", lambda: mock_client)
    monkeypatch.setattr(ct, "obtener_champion_actual", lambda *_: ("4", 0.8293))

    decision = ct.ejecutar_continuous_training(report, "mibici-equipo01-p2-riesgo")

    assert decision.retrained is False
    assert decision.promoted is False
    assert decision.champion_version == "4"
    mock_client.set_registered_model_alias.assert_not_called()


def test_con_drift_promueve_si_gana(tmp_path, monkeypatch):
    report = _write_drift(tmp_path, dataset_drift=True)
    mock_client = MagicMock()
    monkeypatch.setattr(ct, "MlflowClient", lambda: mock_client)
    monkeypatch.setattr(ct, "obtener_champion_actual", lambda *_: ("4", 0.8293))
    monkeypatch.setattr(ct, "reentrenar_y_registrar_candidata", lambda *_: ("5", 0.8410))

    decision = ct.ejecutar_continuous_training(report, "mibici-equipo01-p2-riesgo")

    assert decision.retrained is True
    assert decision.promoted is True
    assert decision.champion_version == "5"
    mock_client.set_registered_model_alias.assert_called_once_with(
        "mibici-equipo01-p2-riesgo", "champion", "5"
    )
    mock_client.set_model_version_tag.assert_any_call(
        "mibici-equipo01-p2-riesgo", "5", "stage", "production"
    )
    mock_client.set_model_version_tag.assert_any_call(
        "mibici-equipo01-p2-riesgo", "4", "stage", "archived"
    )


def test_con_drift_no_promueve_si_pierde(tmp_path, monkeypatch):
    report = _write_drift(tmp_path, dataset_drift=True)
    mock_client = MagicMock()
    monkeypatch.setattr(ct, "MlflowClient", lambda: mock_client)
    monkeypatch.setattr(ct, "obtener_champion_actual", lambda *_: ("4", 0.8293))
    monkeypatch.setattr(ct, "reentrenar_y_registrar_candidata", lambda *_: ("5", 0.8100))

    decision = ct.ejecutar_continuous_training(report, "mibici-equipo01-p2-riesgo")

    assert decision.retrained is True
    assert decision.promoted is False
    assert decision.champion_version == "4"
    mock_client.set_registered_model_alias.assert_not_called()
    mock_client.set_model_version_tag.assert_any_call(
        "mibici-equipo01-p2-riesgo", "5", "stage", "evaluation"
    )


def test_force_retrain_ejecuta_sin_drift(tmp_path, monkeypatch):
    report = _write_drift(tmp_path, dataset_drift=False)
    mock_client = MagicMock()
    monkeypatch.setattr(ct, "MlflowClient", lambda: mock_client)
    monkeypatch.setattr(ct, "obtener_champion_actual", lambda *_: ("4", 0.8293))
    monkeypatch.setattr(ct, "reentrenar_y_registrar_candidata", lambda *_: ("5", 0.8200))

    decision = ct.ejecutar_continuous_training(
        report, "mibici-equipo01-p2-riesgo", force_retrain=True
    )

    assert decision.retrained is True
    assert decision.promoted is False
    assert decision.champion_version == "4"
