"""Verifica que los modelos @champion se resuelven y cargan desde el Model Registry.

Corre igual en el host y dentro del contenedor (issues #124 y #126):
    python scripts/verificar_registry.py
"""

import os
from pathlib import Path
import sys

# Permite importar mibici sin instalar el paquete (en el contenedor PYTHONPATH ya es /app)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow

from mibici.api.registry import _ALIASES, load_champion


def main() -> int:
    print(f"MLFLOW_TRACKING_URI = {os.getenv('MLFLOW_TRACKING_URI') or '(sin definir)'}")
    client = mlflow.MlflowClient()
    fallos = 0
    for key, name in _ALIASES.items():
        try:
            version = client.get_model_version_by_alias(name, "champion")
            model = load_champion(key)
            print(
                f"OK   {key}: {name} v{version.version} run_id={version.run_id} -> {type(model).__name__}"
            )
        except Exception as error:  # noqa: BLE001 - se reporta y se sigue con el siguiente
            fallos += 1
            print(f"FAIL {key}: {name} -> {type(error).__name__}: {error}")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
