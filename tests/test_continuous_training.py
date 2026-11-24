import json
from pathlib import Path
from unittest.mock import MagicMock

from mibici.modeling import continuous_training as ct


def _write_drift(tmp_path: Path, dataset_drift: bool) -> Path:
    path = tmp_path / "drift_report.json"
    path.write_text(
        json.dumps(
            {
                "dataset_drift": dataset_drift,
                "number_of_drifted_columns": 12 if dataset_drift else 4,
                "number_of_columns": 21,
                "share_of_drifted_columns": 0.571 if dataset_drift else 0.190,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_sin_drift_no_reentrena(tmp_path, monkeypatch):
    report = _write_drift(tmp_path, dataset_drift=False)
    mock_client = MagicMock()
    monkeypatch.setattr(ct, "MlflowClient", lambda: mock_client)
    monkeypatch.setattr(ct, "obtener_champion_actual", lambda *_: ("2", 0.8293))

    decision = ct.ejecutar_continuous_training(report, "mibici-equipo01-p2-riesgo")

    assert decision.retrained is False
    assert decision.promoted is False
    assert decision.champion_version == "2"
    mock_client.set_registered_model_alias.assert_not_called()


def test_con_drift_promueve_si_gana(tmp_path, monkeypatch):
    report = _write_drift(tmp_path, dataset_drift=True)
    mock_client = MagicMock()
    monkeypatch.setattr(ct, "MlflowClient", lambda: mock_client)
    monkeypatch.setattr(ct, "obtener_champion_actual", lambda *_: ("2", 0.8293))
    monkeypatch.setattr(ct, "reentrenar_y_registrar_candidata", lambda *_: ("5", 0.8410))

    decision = ct.ejecutar_continuous_training(report, "mibici-equipo01-p2-riesgo")

    assert decision.retrained is True
    assert decision.promoted is True
    assert decision.champion_version == "5"
    mock_client.set_registered_model_alias.assert_called_once_with(
        "mibici-equipo01-p2-riesgo", "champion", "5"
    )


def test_con_drift_no_promueve_si_pierde(tmp_path, monkeypatch):
    report = _write_drift(tmp_path, dataset_drift=True)
    mock_client = MagicMock()
    monkeypatch.setattr(ct, "MlflowClient", lambda: mock_client)
    monkeypatch.setattr(ct, "obtener_champion_actual", lambda *_: ("2", 0.8293))
    monkeypatch.setattr(ct, "reentrenar_y_registrar_candidata", lambda *_: ("5", 0.8100))

    decision = ct.ejecutar_continuous_training(report, "mibici-equipo01-p2-riesgo")

    assert decision.retrained is True
    assert decision.promoted is False
    assert decision.champion_version == "2"
    mock_client.set_registered_model_alias.assert_not_called()
