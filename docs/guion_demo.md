# Guion del video final — Proyecto MiBici, Equipo 01

**Entregable:** video continuo `TC5061_Proyecto_Final_Equipo01.mp4`, **mínimo 2:00 y máximo 5:00** (objetivo **4:30**), sin cortes de edición y grabado por el equipo (issue #200).

**Criterio de rúbrica:** Presentación y demostración (20 pts). Para el nivel *Excelente* el video debe mostrar la **API respondiendo peticiones reales** y comunicar con claridad el **problema**, la **solución** y las **decisiones de MLOps**.

**Material:** presentación `TC5061_Proyecto_Final_Equipo01.pdf` (13 diapositivas, exportada de `reports/Proyecto Final MLOps - MiBici Equipo 01.pptx`) y el sistema levantado con Docker Compose y el Model Registry real.

---

## 1. Reparto de turnos

| Rol | Integrante | Narra |
|---|---|---|
| **Teclado** (único que manipula pantalla y comandos) | **Gerardo** (@laloz60) | Diapositivas 7, 8 y 12; demo de la API y del CI/CD |
| Narrador A — problema y FinOps | **Sebastian** (@smuciton) | Diapositivas 3, 4 y 9; cierre |
| Narrador B — experimentación y Registry | **Jesús** (@JesusTafolla) | Diapositivas 5 y 6; demo del Model Registry |
| Narrador C — gobernanza y decisiones | **Gemma** (@GemmaMartinez) | Diapositivas 1, 2, 10 y 11; demo del reporte de drift |

Cada integrante habla alrededor de un minuto.

**Reglas de grabación**
- Una sola voz a la vez. Cada narrador termina su parte con la frase puente indicada en el guion; esa es la señal para que Gerardo cambie de diapositiva o de ventana.
- Gerardo no habla mientras escribe o hace clic, salvo en sus propios bloques.
- Si alguien se equivoca en una frase, **sigue adelante sin detenerse**: el video no admite cortes.

---

## 2. Preparación antes de grabar

### 2.1 Comandos, en orden (15 minutos antes de grabar)

Todos se ejecutan en la **Terminal** de la Mac de Gerardo. Copiar cada bloque completo.

**Paso 1. Ir al repositorio y actualizar `main`**
```bash
cd ~/Documents/MaestriaTecMonterrey/CuartoTrimestre/MLOps/Semana_5/TC5061-Proyecto-Equipo01
git switch main
git pull --ff-only origin main
```

**Paso 2. Abrir Docker Desktop y esperar a que esté listo**
```bash
open -a Docker
until docker info >/dev/null 2>&1; do sleep 2; done; echo "Docker listo"
```

**Paso 3. Confirmar que los puertos 8000 y 5001 están libres**
```bash
lsof -nP -iTCP:8000 -iTCP:5001 -sTCP:LISTEN
```
Resultado esperado: **ninguna línea**. Si aparece un proceso, detenerlo (por ejemplo, otro backend en el 8000) antes de seguir.

**Paso 4. Levantar el sistema**
```bash
docker compose up -d --build --wait
```
Resultado esperado: termina sin error con `mibici-mlflow-1` y `mibici-api-1` en `Healthy`.

**Paso 5. Verificar el sistema**
```bash
docker compose ps
bash scripts/test_api.sh http://localhost:8000
```
Resultado esperado: los dos servicios `(healthy)`; en `test_api.sh`, `HTTP status: 200` en las pruebas 1 a 4 y `HTTP status: 422` en la 5.

**Paso 6. Abrir las ventanas** (en una ventana nueva del navegador, para que las pestañas queden en el orden de ⌘1 a ⌘4)
```bash
open reports/TC5061_Proyecto_Final_Equipo01.pdf
open "http://localhost:8000/docs"
open "http://localhost:5001/#/models"
open reports/drift_report.html
gh run view "$(gh run list -b main -L 1 --json databaseId --jq '.[0].databaseId')" --web
```

| Ventana | Atajo | Qué se abrió | Dejarla así antes de grabar |
|---|---|---|---|
| **T1** | Vista Previa | `TC5061_Proyecto_Final_Equipo01.pdf` | **Visualización → Pase de diapositivas** (⇧⌘F), en la diapositiva 1; se avanza con → |
| **T2** | Terminal | Esta misma terminal | Letra grande (⌘+ dos o tres veces); se limpia en el paso 7 |
| **T3** | ⌘1 | Swagger | Expandir `POST /predict/p1-demanda` → **Try it out** → pegar el payload del lunes 8:00 (sección 2.4). Lo mismo en `POST /predict/p2-riesgo`. En `POST /predict/p3-clustering` → **Try it out** (dejar el ejemplo de oficina) |
| **T4** | ⌘2 | MLflow UI, **Models** | Abrir `mibici-equipo01-p2-riesgo`, donde se ven la versión 2 (`champion`) y la 4 (`ab-candidate`) |
| **T5** | ⌘3 | Reporte de Evidently | Al inicio del resumen de drift |
| **T6** | ⌘4 | Último run de `main` en GitHub Actions | Página **Summary** con el grafo de jobs en verde |

Zoom del navegador a **90 %** (⌘ y −) para que se vean completos el grafo del CI y las respuestas de Swagger.

**Paso 7. Revisión de 30 segundos, justo antes de presionar grabar**
```bash
docker compose ps
curl -s localhost:8000/health
curl -s -o /dev/null -w "%{http_code}\n" localhost:5001/
clear
```
Resultado esperado: ambos `(healthy)`; `{"status":"healthy","modelos_cargados":["p1","p2","p3"]}`; `200`. El `clear` deja la terminal vacía para el bloque D1.

### 2.2 Comandos durante la grabación

Solo hay **un** comando en todo el video (bloque D1, en T2):
```bash
docker compose ps
```
Todo lo demás son clics en Swagger, MLflow, Evidently y GitHub Actions.

**Solo en caso de emergencia** (si Swagger no responde, ver sección 5):
```bash
bash scripts/test_api.sh http://localhost:8000
```

### 2.3 Comandos después de grabar

```bash
docker compose down
```
Apaga los contenedores; `mlflow.db` y `mlruns/` se quedan en la carpeta del proyecto.

### 2.4 Payloads (JSON) para Swagger

El *payload* es el cuerpo JSON de la petición: lo que se pega en el cuadro **Request body** de Swagger. Se pegan **antes de grabar** (paso 6); durante el video solo se da clic en **Execute**. Tener una copia en otro dispositivo o impresa, nunca en la pantalla que se graba.

**Cómo pegar un payload en Swagger**
1. Dar clic en el endpoint para expandirlo y luego en **Try it out**.
2. Dar clic dentro del cuadro **Request body**, seleccionar todo (⌘A) y borrarlo.
3. Pegar el JSON (⌘V). **No dar Execute** hasta la grabación.

Respuestas verificadas con el sistema levantado y los `@champion` v2 del Model Registry.

#### Payload A — P1 y P2: lunes 8:00, estación con alta demanda reciente

Se pega **el mismo JSON** en `POST /predict/p1-demanda` y en `POST /predict/p2-riesgo` (los dos modelos usan las mismas 15 variables).

```json
{"hora_del_dia": 8, "dia_semana": 0, "mes": 10, "es_fin_de_semana": 0,
 "salidas_lag_1h": 12.0, "salidas_lag_24h": 14.0, "salidas_lag_168h": 13.0,
 "salidas_media_7d": 11.5, "salidas_prom_4sem": 12.0,
 "neto_lag_1h": -6.0, "neto_lag_24h": -5.0, "neto_media_7d": -3.0,
 "capacidad": 20.0, "latitud": 20.6736, "longitud": -103.3440}
```

| Campo | Valor | Significado |
|---|---|---|
| `hora_del_dia`, `dia_semana`, `mes`, `es_fin_de_semana` | 8, 0, 10, 0 | Lunes (0 = lunes) a las 8:00, en octubre, entre semana |
| `salidas_lag_1h`, `salidas_lag_24h`, `salidas_lag_168h` | 12, 14, 13 | Bicis que salieron hace 1 hora, a la misma hora ayer y hace una semana |
| `salidas_media_7d`, `salidas_prom_4sem` | 11.5, 12.0 | Promedios recientes de salidas: estación con mucha demanda |
| `neto_lag_1h`, `neto_lag_24h`, `neto_media_7d` | −6, −5, −3 | Salen más bicis de las que llegan: la estación se está vaciando |
| `capacidad`, `latitud`, `longitud` | 20, 20.6736, −103.3440 | Estación de 20 anclajes en el centro de Guadalajara |

Respuestas esperadas (código **200**):
```json
{"demanda_predicha_viajes": 12.897289839069314}
```
```json
{"riesgo_desabasto_o_saturacion": 1, "probabilidad_riesgo": 0.9994939017750084}
```
Se narra como "cerca de 13 viajes" y "en riesgo, con 99.9 %".

#### Payload B — P3: perfil horario de una estación tipo oficina

En `POST /predict/p3-clustering`. **Es el ejemplo que ya trae Swagger**: basta con **Try it out** y dejarlo como está.

```json
{"perfil_horario": [0,0,0,0,0,1,4,12,20,10,6,5,6,6,5,6,9,15,18,9,5,3,1,0]}
```

Son los viajes en cada hora del día, de las 0 a las 23 h: picos a las 8 h (20 viajes) y a las 18 h (18 viajes), típico de una zona de oficinas. La API lo convierte en proporciones, igual que en el entrenamiento.

Respuesta esperada (código **200**):
```json
{"cluster_asignado": 1}
```

#### Payload C — P3: entrada inválida, para mostrar el 422

En el mismo `POST /predict/p3-clustering`, **durante la grabación** y después de ejecutar el payload B: reemplazar el contenido del cuadro por este JSON y dar **Execute** otra vez.

```json
{"perfil_horario": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
```

Un perfil sin viajes no se puede normalizar, así que Pydantic lo rechaza. Respuesta esperada (código **422**), con el mensaje:
```text
Value error, el perfil horario debe tener al menos un viaje (suma mayor que 0)
```

---

## 3. Línea de tiempo (objetivo 4:30, límite 5:00)

| # | Pantalla | Duración | Acumulado | Narra |
|---|---|---:|---:|---|
| 1 | Diapositiva 1 · Portada | 0:10 | 0:10 | Gemma |
| 2 | Diapositiva 2 · Agenda | 0:05 | 0:15 | Gemma |
| 3 | Diapositiva 3 · Escala del Problema y Datos | 0:15 | 0:30 | Sebastian |
| 4 | Diapositiva 4 · Tres Preguntas de Negocio | 0:15 | 0:45 | Sebastian |
| 5 | Diapositiva 5 · Ingeniería y MLflow Project | 0:15 | 1:00 | Jesús |
| 6 | Diapositiva 6 · Model Registry y Linaje | 0:17 | 1:17 | Jesús |
| 7 | Diapositiva 7 · Arquitectura y Pipeline CI/CD | 0:16 | 1:33 | Gerardo |
| 8 | Diapositiva 8 · Pruebas de Carga con Locust | 0:10 | 1:43 | Gerardo |
| 9 | Diapositiva 9 · Optimización FinOps en AWS | 0:11 | 1:54 | Sebastian |
| 10 | Diapositiva 10 · Drift, A/B Testing y Gobernanza | 0:16 | 2:10 | Gemma |
| 11 | Diapositiva 11 · Decisiones de MLOps | 0:21 | 2:31 | Gemma |
| 12 | Diapositiva 12 · Demo en Vivo y Conclusiones | 0:04 | 2:35 | Gerardo |
| D1 | T2 · `docker compose ps` | 0:10 | 2:45 | Gerardo |
| D2 | T3 · Swagger P1 | 0:20 | 3:05 | Gerardo |
| D3 | T3 · Swagger P2 | 0:10 | 3:15 | Gerardo |
| D4 | T3 · Swagger P3 y 422 | 0:15 | 3:30 | Gerardo |
| D5 | T4 · MLflow UI, Model Registry | 0:30 | 4:00 | Jesús |
| D6 | T5 · Reporte de Evidently | 0:13 | 4:13 | Gemma |
| D7 | T6 · GitHub Actions en verde | 0:12 | 4:25 | Gerardo |
| C | T1 · Diapositiva 12 · Cierre | 0:10 | **4:35** | Sebastian |

La diapositiva 13 (fuentes de imágenes) no se muestra en el video.

**Si el ensayo pasa de 4:45:** acortar la diapositiva 8 (decir solo la conclusión), la 9 (solo el −75 %) y saltar el 422 en D4. **Si queda por debajo de 3:30:** no agregar contenido; hablar más despacio.

---

## 4. Guion detallado

> Las frases en *cursiva* son la señal para que Gerardo avance a la siguiente pantalla.

### Parte 1 — Presentación (0:00 – 2:35)

**Diapositiva 1 · Portada (0:10) — Gemma**
> "Hola, somos el Equipo 01 de Operaciones de Aprendizaje Automático. Les presentamos el proyecto integrador: un sistema de MLOps completo para MiBici, la bicicleta pública de Guadalajara."

*Gerardo avanza.*

**Diapositiva 2 · Agenda (0:05) — Gemma**
> "Veremos el problema, la solución, las decisiones y la demo."

*Gerardo avanza.*

**Diapositiva 3 · Escala del Problema y Datos (0:15) — Sebastian**
> "Trabajamos con dieciséis millones de viajes de 470 estaciones en 42 meses, de mayo de 2023 a octubre de 2026. Cada archivo se verifica con su hash, y la partición es temporal: entrenamos con el pasado y evaluamos con el futuro."

*Gerardo avanza.*

**Diapositiva 4 · Tres Preguntas de Negocio (0:15) — Sebastian**
> "Respondemos tres preguntas. P1: cuántos viajes saldrán de cada estación, con un Random Forest. P2: qué estaciones están en riesgo de quedarse sin bicis o sin anclajes, con regresión logística. Y P3: qué estaciones comparten patrones de uso, con K-Means++."

*"…y para que esto sea reproducible…" → Gerardo avanza.*

**Diapositiva 5 · Ingeniería y MLflow Project (0:15) — Jesús**
> "El repositorio sigue Cookiecutter y el pipeline completo es un MLflow Project: con `mlflow run` se ingiere, se generan features, se entrena y se registra desde un clon limpio. Además, 102 pruebas automáticas y chequeos de calidad y seguridad."

*Gerardo avanza.*

**Diapositiva 6 · Model Registry y Linaje (0:17) — Jesús**
> "El mejor modelo de cada pregunta está registrado en MLflow con el alias `champion`, elegido por su métrica de validación, y guarda su linaje: run, commit y hashes de los datos. La API carga el `champion` al arrancar; promoverlo no exige reconstruir la imagen."

*"…y así lo desplegamos." → Gerardo avanza.*

**Diapositiva 7 · Arquitectura y Pipeline CI/CD (0:16) — Gerardo**
> "Docker Compose levanta MLflow y la API con un solo comando. En cada cambio, GitHub Actions revisa el código, corre las pruebas en Ubuntu y macOS, publica la imagen, prueba el sistema completo y deja un resumen en el pull request."

*Gerardo avanza.*

**Diapositiva 8 · Pruebas de Carga con Locust (0:10) — Gerardo**
> "Con Locust probamos tres escenarios. Con dos réplicas detrás de Nginx, la latencia p95 baja hasta 89 %; el cuello de botella es la CPU."

*"Sebastian, ¿y cuánto cuesta?" → Gerardo avanza.*

**Diapositiva 9 · Optimización FinOps en AWS (0:11) — Sebastian**
> "Hoy cuesta entre 53 centavos y 1.55 dólares por millón de inferencias. Separar MLflow, que ocupa casi 2 GB, baja el costo hasta 75 %."

*Gerardo avanza.*

**Diapositiva 10 · Drift, A/B Testing y Gobernanza (0:16) — Gemma**
> "Con Evidently, 4 de 21 variables cambiaron, pero el conjunto no cruza el umbral de drift. En el A/B de P2, la versión reentrenada no mejora por sí misma: la diferencia viene del umbral. Todo está en el Model Card."

*Gerardo avanza.*

**Diapositiva 11 · Decisiones de MLOps (0:21) — Gemma**
> "Nuestras decisiones clave: partición temporal y selección por validación, para una evaluación honesta; reentrenar solo con drift global, con un ciclo de Continuous Training que promueve la nueva versión solo si gana; carga dinámica del `champion`; el mismo preprocesamiento al entrenar y al servir; y separar MLflow para bajar costos."

*"Ahora, veámoslo funcionando." → Gerardo avanza.*

**Diapositiva 12 · Demo en Vivo y Conclusiones (0:04) — Gerardo**
> "Vamos a la demo."

*Gerardo cambia a T2.*

### Parte 2 — Demo funcional (2:35 – 4:25)

**D1 · Sistema levantado (0:10) — Gerardo**
- **Acción:** en T2 escribe `docker compose ps` y presiona Enter.
> "El sistema está levantado con Docker Compose: MLflow y la API, ambos en estado *healthy*, conectados al Model Registry real del equipo."

*Gerardo cambia a T3.*

**D2 · Swagger, P1 (0:20) — Gerardo**
- **Acción:** en `POST /predict/p1-demanda`, que ya tiene pegado el **payload A**, da clic en **Execute** y baja hasta **Responses**.
- **Payload A** (ya pegado desde la preparación):
  ```json
  {"hora_del_dia": 8, "dia_semana": 0, "mes": 10, "es_fin_de_semana": 0,
   "salidas_lag_1h": 12.0, "salidas_lag_24h": 14.0, "salidas_lag_168h": 13.0,
   "salidas_media_7d": 11.5, "salidas_prom_4sem": 12.0,
   "neto_lag_1h": -6.0, "neto_lag_24h": -5.0, "neto_media_7d": -3.0,
   "capacidad": 20.0, "latitud": 20.6736, "longitud": -103.3440}
  ```
- **En pantalla:** código **200** y `{"demanda_predicha_viajes": 12.897…}`.
> "Hacemos una petición real: un lunes a las 8 de la mañana, en una estación con mucha actividad reciente. El modelo predice cerca de 13 viajes en la siguiente hora, con código 200."

**D3 · Swagger, P2 (0:10) — Gerardo**
- **Acción:** en `POST /predict/p2-riesgo`, que ya tiene pegado **el mismo payload A**, da clic en **Execute** y baja hasta **Responses**.
- **En pantalla:** código **200** y `{"riesgo_desabasto_o_saturacion": 1, "probabilidad_riesgo": 0.9994…}`.
> "Para la misma estación, P2 la marca en riesgo de desabasto, con probabilidad de 99.9 %."

**D4 · Swagger, P3 y entrada inválida (0:15) — Gerardo**
- **Acción 1:** en `POST /predict/p3-clustering`, con el **payload B** (el ejemplo de oficina que ya trae Swagger), da clic en **Execute**.
  ```json
  {"perfil_horario": [0,0,0,0,0,1,4,12,20,10,6,5,6,6,5,6,9,15,18,9,5,3,1,0]}
  ```
- **En pantalla:** código **200** y `{"cluster_asignado": 1}`.
- **Acción 2:** en el mismo cuadro, selecciona todo (⌘A), pega el **payload C** y da clic en **Execute** otra vez.
  ```json
  {"perfil_horario": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
  ```
- **En pantalla:** código **422** y el mensaje "el perfil horario debe tener al menos un viaje".
> "P3 asigna esta estación tipo oficina al clúster 1; la API normaliza el perfil igual que en el entrenamiento. Y si mandamos un perfil sin viajes, Pydantic lo rechaza con un 422."

*"Jesús, ¿de dónde vienen estos modelos?" → Gerardo cambia a T4.*

**D5 · MLflow UI, Model Registry (0:30) — Jesús**
- **Acción:** Gerardo abre `mibici-equipo01-p2-riesgo` y señala con el cursor los aliases y los tags de las versiones 2 y 4.
> "Este es el Model Registry. Los tres modelos tienen su versión 2 con el alias `champion` y el tag `stage=production`: es la que acaba de responder la API. En P2, la versión 4 es la `ab-candidate`, con `stage=evaluation`: la candidata del A/B, fuera de producción porque no superó al `champion`. Cada versión guarda la métrica con la que se eligió."

*Gerardo cambia a T5.*

**D6 · Reporte de Evidently (0:13) — Gemma**
- **Acción:** Gerardo muestra el resumen del reporte de drift.
> "Este reporte de Evidently se regenera con `mlflow run . -e drift`. Hay 4 de 21 variables con drift, por debajo del umbral global, así que el Continuous Training no reentrena."

*Gerardo cambia a T6.*

**D7 · GitHub Actions en verde (0:12) — Gerardo**
- **Acción:** muestra el grafo del último run de `main`.
> "Y este es el pipeline de CI/CD en la rama principal: calidad, pruebas en Ubuntu y macOS, construcción de la imagen, prueba de humo con Docker Compose y notificación. Todo en verde."

*Gerardo regresa a T1, diapositiva 12.*

### Cierre (4:25 – 4:35)

**C · Cierre (0:10) — Sebastian**
> "Así llevamos un modelo de la idea a un sistema operable: reproducible, registrado, desplegado, automatizado y monitoreado. Gracias."

*Gerardo detiene la grabación.*

---

## 5. Contingencias durante la grabación

| Si pasa esto | Hacer esto (sin detener el video) |
|---|---|
| Swagger tarda en responder | Esperar en silencio; en el último caso, correr `bash scripts/test_api.sh` en T2 y narrar sobre esa salida |
| La UI de MLflow no carga o responde 403 | Mostrar la tabla de la diapositiva 6 y continuar; antes de repetir la toma, revisar `MLFLOW_SERVER_ALLOWED_HOSTS` en `docker-compose.yml` |
| Alguien se traba en su texto | Seguir con la siguiente frase; no regresar |
| El cronómetro marca 4:45 antes de D7 | Saltar D7 e ir directo al cierre |

---

## 6. Ensayo

- [ ] **Ensayo 1:** sin narración, solo los clics y cambios de ventana, para medir tiempos de carga.
- [ ] **Ensayo 2:** completo, con cronómetro. Objetivo entre 4:15 y 4:45.
- [ ] Verificar que en pantalla **no** aparecen `.env`, tokens, notificaciones del sistema ni archivos personales (activar "No molestar").
- [ ] Verificar el audio: un solo micrófono o una sola voz a la vez, sin eco.

## 7. Después de grabar (#200)

1. Revisar que el video dure entre 2:00 y 5:00 y que no tenga cortes.
2. Exportarlo como `TC5061_Proyecto_Final_Equipo01.mp4`.
3. Subirlo a Canvas junto con `TC5061_Proyecto_Final_Equipo01.pdf`.
4. Agregar la liga al video en la diapositiva 12 de la presentación (#197) y en el README (#189).

---

**Relacionado:** #194 (despliegue verificado) · #196 y #197 (presentación) · #199 (Continuous Training) · #200 (grabación).
