# Model Registry: linaje por validación y estado de producción

Evidencia del issue #191. Ensayo en un registry local (`mlflow.db` + `mlruns/`, servidor en `http://127.0.0.1:5000`), el 24 nov 2026, rama `feature/issue-191-registry`.

## Cambios en `mibici/modeling/register.py`

- La versión elegida se marca con el alias `@champion` y los tags `stage=production`, `pregunta`, `metrica_seleccion` y `valor_metrica`.
- La versión que deja de ser `@champion` queda con `stage=archived`; no se borra.
- La versión con alias `@ab-candidate` (retador del A/B test de P2) queda con `stage=evaluacion` y una descripción que indica que no sirve tráfico.
- Si el mejor run ya está registrado, no crea otra versión: completa descripción, tags y alias. Re-ejecutar es seguro.
- La selección omite las corridas sin la métrica de validación. Sin este filtro, P3 elegía `p3-candidate-k6`, una corrida anterior a las métricas `val_*`, y el registro fallaba al formatear la métrica vacía.

## Comando

```bash
mlflow run . --experiment-name mibici-equipo01 --env-manager uv -e register
```

```
mibici-equipo01-p1-demanda v2 deja @champion: archived
mibici-equipo01-p1-demanda v3 <- p1-candidate-rf-n200 (val_mae=0.8079)
mibici-equipo01-p2-riesgo v2 deja @champion: archived
mibici-equipo01-p2-riesgo v3 (@ab-candidate) marcado como evaluacion
mibici-equipo01-p2-riesgo v4 <- P2-parent-search-logreg (val_roc_auc=0.8293)
mibici-equipo01-p3-grupos v2 deja @champion: archived
mibici-equipo01-p3-grupos v3 <- p3-candidate-k5 (val_silhouette_score=0.3268)
=== Run (ID '392d1a60ce0c440e97b5b7eb4ff4e55c') succeeded ===
```

## Estado del registry

| Modelo | Versión | Alias | `stage` | Run de origen | Métrica de selección |
|---|---:|---|---|---|---|
| `mibici-equipo01-p1-demanda` | 1 | — | — | `P1-candidato-rf` (`dcb88cd8`) | — |
| `mibici-equipo01-p1-demanda` | 2 | — | archived | `p1-candidate-rf-n200` (`41e42807`) | — |
| `mibici-equipo01-p1-demanda` | 3 | `@champion` | production | `p1-candidate-rf-n200` (`e2ee147a`) | val_mae=0.8079 |
| `mibici-equipo01-p2-riesgo` | 1 | — | — | `P2-parent-search-logreg` (`eb0ce972`) | — |
| `mibici-equipo01-p2-riesgo` | 2 | — | archived | `P2-parent-search-logreg` (`3ad75953`) | — |
| `mibici-equipo01-p2-riesgo` | 3 | `@ab-candidate` | evaluacion | `P2-v2-ventana-extendida` (`e67f43d2`) | — |
| `mibici-equipo01-p2-riesgo` | 4 | `@champion` | production | `P2-parent-search-logreg` (`3d8fdf6a`) | val_roc_auc=0.8293 |
| `mibici-equipo01-p3-grupos` | 1 | — | — | `p3-baseline-k5` (`93caace1`) | — |
| `mibici-equipo01-p3-grupos` | 2 | — | archived | `p3-candidate-k3` (`9fa2a199`) | — |
| `mibici-equipo01-p3-grupos` | 3 | `@champion` | production | `p3-candidate-k5` (`ab2c73d6`) | val_silhouette_score=0.3268 |

Las v1 dejaron de ser `@champion` antes de este cambio, por eso no tienen tag `stage`.

Descripción de linaje de un `@champion` (P2 v4):

> Modelo elegido de P2 sobre el experimento 'mibici-equipo01'. Run de origen: P2-parent-search-logreg (3d8fdf6a1e344d329a82220d9796e13f). Commit: 97b7e22236b4a974e4a8d0b320026dca47b07822. Datos: manifest 7db3fa3702f6b57f / parquet ed2f4783e3a3ec77. val_roc_auc=0.8293. Registrado con mibici/modeling/register.py.

## Verificación de carga

```bash
python scripts/verificar_registry.py
```

```
MLFLOW_TRACKING_URI = http://127.0.0.1:5000
OK   p1: mibici-equipo01-p1-demanda v3 run_id=e2ee147ac0a84f95bb85c28bd4bf3c92 -> PyFuncModel
OK   p2: mibici-equipo01-p2-riesgo v4 run_id=3d8fdf6a1e344d329a82220d9796e13f -> PyFuncModel
OK   p3: mibici-equipo01-p3-grupos v3 run_id=ab2c73d64f3049f9b53128d1de0f379b -> PyFuncModel
```

## La API con los nuevos `@champion`

Payloads de `scripts/test_api.sh`, contra `mibici.api.main` con `TestClient` apuntando al registry local:

| Petición | HTTP | Respuesta |
|---|---:|---|
| `GET /health` | 200 | `modelos_cargados: [p1, p2, p3]` |
| `POST /predict/p1-demanda` | 200 | `demanda_predicha_viajes: 5.574` |
| `POST /predict/p2-riesgo` | 200 | `riesgo_desabasto_o_saturacion: 1`, `probabilidad_riesgo: 0.744` |
| `POST /predict/p3-clustering` | 200 | `cluster_asignado: 2` |
| Payload inválido | 422 | Validación de Pydantic |

El cambio de P3 de k=3 a k=5 no altera el contrato de la API: la entrada sigue siendo el perfil horario de 24 valores.

## Pruebas

- `pytest tests/test_register.py`: 19 pasan (6 nuevas para tags, archivado, `@ab-candidate`, versiones ya registradas y corridas sin `val_*`).
- `pytest`: 88 de 89. La que falla es `test_unified_parquet_exists_and_is_non_empty` por la diferencia conocida de 26 viajes del CSV 2023-05 (ver README, "Descarga de datos"); no se relaciona con este cambio.
- `ruff check .`: sin observaciones.

## Capturas

UI de MLflow (versiones, aliases, descripción y tags) en `reports/figures/registry_issue191/`.
