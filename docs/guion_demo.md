# Guion de la demo (issue #198)

**Entregable:** video continuo `TC5061_Proyecto_Final_Equipo01.mp4`, **mínimo 2 y máximo 5 minutos** (objetivo: **~4:00**), sin cortes de edición, grabado por el equipo.

**Criterio de rúbrica:** Presentación y demostración — 20 pts. La demo funcional (API respondiendo peticiones reales) y una presentación que comunique problema, solución y decisiones de MLOps son lo que separa el nivel *Excelente* del *Satisfactorio*.

---

## 1. Reparto de turnos

| Rol | Integrante | Responsabilidad |
|---|---|---|
| **Teclado** (único que manipula) | **Gerardo (@laloz60)** | Maneja todas las pestañas y comandos; no narra mientras escribe. Dueño de la infra (issue #194). |
| Narrador A — problema y solución | **smuciton (@smuciton)** | Bloques 1a–1b (presentación parte 1, #196). |
| Narrador B — decisiones de MLOps | **Gemma (@GemmaMartinez)** | Bloque 1c + bloque 5 (decisiones, gobernanza, drift; presentación parte 2, #197). |
| Narrador C — modelos y registry | **Jesús (@JesusTafolla)** | Bloques 3–4 (modelos, A/B, Model Registry). |

> Regla de grabación: **una sola voz a la vez**; el teclado espera la señal del narrador (`…y ahora vamos a…`) antes de cambiar de ventana.

---

## 2. Pestañas y comandos preparados ANTES de grabar

Abrir en este orden (nada se abre durante la grabación):

| # | Pestaña / ventana | Contenido |
|---|---|---|
| T1 | Presentación (PDF a pantalla completa) | `reports/TC5061_Proyecto_Final_Equipo01.pdf` (#197) |
| T2 | Terminal A | Sistema ya levantado (ver abajo) |
| T3 | Navegador — Swagger | <http://localhost:8000/docs> |
| T4 | Navegador — MLflow UI | <http://localhost:5001> → modelos `mibici-equipo01-*` |
| T5 | Navegador — Evidently | `reports/drift_report.html` (abierto con `open reports/drift_report.html`) |
| T6 | Navegador — GitHub Actions | Run en verde de `main` |

**Sistema levantado y verificado antes de grabar** (no construir imágenes durante la grabación):

```bash
docker compose up -d --build --wait   # ~75 s en frío; deja ambos healthy
docker compose ps                     # api y mlflow healthy
bash scripts/test_api.sh http://localhost:8000   # 5/5 en verde (ensayo previo)
```

**Pre-chequeo de 30 s antes de grabar:**

```bash
docker compose ps                                  # ¿healthy?
curl -s localhost:8000/health                      # {"status":"healthy",...}
curl -s -o /dev/null -w "%{http_code}\n" localhost:5001/   # 200 (si da 403, falta el fix de ALLOWED_HOSTS)
```

> El bloque 4 depende de la UI de MLflow en `localhost:5001`. Si responde **403** (`Invalid Host header`), el `MLFLOW_SERVER_ALLOWED_HOSTS` de `docker-compose.yml` no incluye el 5001: corregirlo y `docker compose up -d mlflow` (ver hallazgo A de `reports/api/evidencia_despliegue_final.md`).

---

## 3. Secuencia cronometrada (objetivo 4:00)

| Bloque | Tiempo | Acumulado | Quién narra | Qué se muestra |
|---|---:|---:|---|---|
| 0. Portada / título del equipo | 0:10 | 0:10 | Gemma | T1: portada del PDF |
| 1. Problema, datos y arquitectura | 1:10 | 1:20 | smuciton (1a–1b) | T1: diapositivas de problema + arquitectura |
| 2. Decisiones de MLOps (resumen) | 0:40 | 2:00 | Gemma | T1: diapositiva de decisiones |
| 3. Sistema levantado + API real | 0:50 | 2:50 | Gerardo (voz guía) | T2 `docker compose ps` → T3 Swagger: petición real a P1, P2 y P3 |
| 4. Model Registry y A/B | 0:45 | 3:35 | Jesús | T4 MLflow UI: versiones, `@champion`, `@ab-candidate` |
| 5. Monitoreo (drift) y gobernanza | 0:35 | 4:10 | Gemma | T5 reporte Evidently: 4/21 columnas con drift, dataset sin drift |
| 6. CI/CD en verde + cierre | 0:35 | 4:45 | Gerardo | T6 run de GitHub Actions verde → cierre de Jesús/Gemma |

**Límite duro: 5:00.** Si el ensayo cronometrado pasa de 4:45, recortar del bloque 1 (dejar 1 sola diapositiva de arquitectura) o del bloque 5 (mostrar solo el resumen, no las gráficas).

---

## 4. Texto y acciones por bloque

### Bloque 0 — Portada (0:10)
- **Gemma:** "Somos el Equipo 01 de TC5061. Les mostramos el sistema completo de MLOps de MiBici: de los datos al modelo en producción, en menos de cinco minutos."

### Bloque 1 — Problema, datos, arquitectura (1:10)
- **smuciton:**
  - *Problema (0:25):* MiBici tiene ~470 estaciones; queremos **P1** predecir la demanda de viajes, **P2** anticipar estaciones en riesgo de quedarse sin bicis/anclajes, y **P3** agrupar estaciones por patrón de uso.
  - *Datos (0:20):* 2023-05 a 2026-10, **partición temporal sin fuga** (train/val/test por fechas).
  - *Arquitectura (0:25):* "Cookiecutter → feature engineering → MLflow Project/Tracking → Registry → API en Docker Compose → CI/CD con GitHub Actions → Locust → monitoreo con Evidently."

### Bloque 2 — Decisiones de MLOps (0:40)
- **Gemma:** elegimos **MLflow Project** para reproducir el pipeline con un solo comando (`mlflow run . -e train`); **selección por validación** (`val_mae`, `val_roc_auc`, `val_silhouette_score`) para no mirar el test; **Registry con aliases** `@champion` para que la API cargue la versión buena y no una candidata; **A/B testing** de P2 antes de promover; **CI/CD** que corre quality, tests en matriz ubuntu/macos, build Docker, smoke test y notificación.

### Bloque 3 — Sistema levantado + API real (0:50) ⭐ el bloque que exige la rúbrica
- **Gerardo:** "El sistema está corriendo con el Model Registry real, no el simulado del CI."
  1. T2: `docker compose ps` → ambos **healthy**.
  2. T3 (Swagger): "Petición real a P1" → payload de ejemplo → **Execute** → `200` con `demanda_predicha_viajes`.
  3. "Lo mismo para P2" → `probabilidad_riesgo` → **200**.
  4. "Y P3, que normaliza el perfil horario como en el entrenamiento" → `cluster_asignado` → **200**.
  5. *(opcional si hay tiempo)* payload inválido → **422** con la validación de Pydantic.
- **Gerardo (cierre del bloque):** "Cuatro endpoints, respuestas reales del modelo registrado."

### Bloque 4 — Model Registry y A/B (0:45)
- **Jesús:** T4 MLflow UI → `mibici-equipo01-p1-demanda`:
  - "Tres modelos registrados y versionados. La versión **2** de P1 está como **`@champion`** — es la que sirve la API."
  - "P2 también tiene `@champion`, y la versión **4** está como **`@ab-candidate`**: es la candidata del A/B testing, **en evaluación, no en producción**."
  - "El linaje está documentado: run de origen, commit y métricas de validación."

### Bloque 5 — Drift y gobernanza (0:35)
- **Gemma:** T5 Evidently → "Comparamos referencia vs. actual, 30 mil filas cada una: **4 de 21 columnas con drift (19 %), el dataset NO marca drift** (el umbral es 50 %). Uno real: `estacion_id`, porque 14 estaciones se dieron de baja. El reporte se regenera con `mlflow run . -e drift`."
- *(Si cabe:)* "El **Model Card** (`docs/MODEL_CARD.md`) documenta métricas, limitaciones y responsables, y el análisis **FinOps** muestra ~0.5 USD por millón de inferencias en spot."

### Bloque 6 — CI/CD verde + cierre (0:35)
- **Gerardo:** T6 → "El último run en `main` está en **verde**: quality, tests en ubuntu y macos, build de Docker, smoke test de la API."
- **Jesús (cierre):** "Modelo registrado, API contenerizada, CI/CD automatizado, monitoreo y gobernanza: el ciclo completo de MLOps, funcionando."

---

## 5. Ensayo cronometrado

- [ ] Ensayo 1: sin narración, solo medir tiempos de carga de ventanas (T3 Swagger puede tardar ~2 s en renderizar).
- [ ] Ensayo 2: completo con narración, cronómetro; objetivo **3:30–4:30**.
- [ ] Verificar que **no** aparecen en pantalla: `.env`, tokens, traceback, o el `~$…docx` de Word.
- [ ] Verificar audio: una sola voz, sin eco de dos narradores.

## 6. Al terminar la grabación (#200)

1. Exportar como `TC5061_Proyecto_Final_Equipo01.mp4` (2–5 min).
2. Subirlo a Canvas y enlazarlo en el PDF de la presentación (#197) y en el README (#189).

---

**Dependencias:** #194 (sistema verificado, PR #206) · #196 y #197 (presentación) · #200 (grabación).
