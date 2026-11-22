# Resumen de PRs del equipo

La tabla reúne los PRs del equipo del 15 al 22 de noviembre de 2026 y enlaza sus evidencias disponibles. (16 de noviembre ASUETO)

## Resumen por integrante

| Nombre | Creó | Revisó |
|---|---|---|
| @JesusTafolla | #170 — Docker Compose; #171 — formato Ruff; #172 — issues 158, 159 y 167; #177 — corrección del tag de CI | #155 — revisión y comentario; #170 — aprobación y comentario; #171 — aprobación y comentario |
| @laloz60 | #155 — entrega de Actividad 10; #170 — Docker Compose; #171 — formato Ruff | #172 — aprobación y comentario; #179 — aprobación y comentario; #180 y #181 — aprobación y comentario |
| @smuciton | #173 — monitoreo con docker stats; #175 — FinOps; #174 — costo por inferencia; #178 — evidencias de carga; #183 — costo por LCU del ALB | #154 — aprobación y comentario; #176 — aprobación y comentario; #177 — aprobación y comentario |
| @GemmaMartinez | #154 — reportes de PRs y revisiones; #176 — pruebas de carga con Locust; #179 — escenarios de carga y gráficas; #180 y #181 — actualización del README | #155 — reporte de Actividad 10 (aprobó y comentó); #173 — docker stats (aprobó y comentó); #175 — FinOps (aprobó y comentó); #183 — revisión y aprobación |

## Resumen por PR

| PR/Commit | Título | Autor | Aprobado | Comentado | Fecha de creación | Fecha de merge | PDF | Screenshot |
|---|---|---|---|---|---|---|---|---|
| #154<br>`8b79b3a` | docs(pr-team): actualizar reportes de PRs y revisiones de la semana 8 al 15 de nov | @GemmaMartinez | @smuciton | @smuciton | 15/11/2026 | 15/11/2026 | [PR_154.pdf](pdf/PR_154.pdf) | [pr_154.png](screenshot/pr_154.png) |
| #155<br>`86919e8` | docs(reporte): entrega Actividad 10 — reporte drift/A-B en PDF y Word | @laloz60 | @GemmaMartinez<br>@JesusTafolla | @GemmaMartinez<br>@JesusTafolla | 15/11/2026 | 18/11/2026 | [PR_155.pdf](pdf/PR_155.pdf) | [pr_155.png](screenshot/pr_155.png) |
| #170<br>`635c521` | feat(infra): orquestación con Docker Compose — API + MLflow con un solo comando | @laloz60 | @JesusTafolla | @JesusTafolla | 18/11/2026 | 18/11/2026 | [PR_170.pdf](pdf/PR_170.pdf) | [pr_170.png](screenshot/pr_170.png) |
| #171<br>`9068fbc` | chore(quality): ruff format en los 3 archivos que dejaban el lint en rojo | @laloz60 | @JesusTafolla | @JesusTafolla | 18/11/2026 | 18/11/2026 | [PR_171.pdf](pdf/PR_171.pdf) | [pr_171.png](screenshot/pr_171.png) |
| #172<br>`274de2f` | Feature/issue 158 159 167 | @JesusTafolla | @laloz60 | @laloz60 | 18/11/2026 | 18/11/2026 | [PR_172.pdf](pdf/PR_172.pdf) | [pr_172.png](screenshot/pr_172.png) |
| #173<br>`97fcb37` | feat(carga): monitoreo de recursos con docker stats en pruebas de carga | @smuciton | @GemmaMartinez | @GemmaMartinez | 19/11/2026 | 21/11/2026 | [PR_173.pdf](pdf/PR_173.pdf) | [pr_173.png](screenshot/pr_173.png) |
| #175<br>`6c7a85b` | docs(finops): optimizaciones viables justificadas con datos de la prueba | @smuciton | @GemmaMartinez | @GemmaMartinez | 19/11/2026 | 21/11/2026 | [PR_175.pdf](pdf/PR_175.pdf) | [pr_175.png](screenshot/pr_175.png) |
| #176<br>`eb038b4` | perf(load): Locust load test suite and reports (ISSUE #160) | @GemmaMartinez | @smuciton | @smuciton | 19/11/2026 | 19/11/2026 | [PR_176.pdf](pdf/PR_176.pdf) | [pr_176.png](screenshot/pr_176.png) |
| #177<br>`85f38ea` | fix(ci): elimina tag sin prefijo para evitar push a Docker Hub en main | @JesusTafolla | @smuciton | @smuciton | 19/11/2026 | 19/11/2026 | [PR_177.pdf](pdf/PR_177.pdf) | [pr_177.png](screenshot/pr_177.png) |
| #179<br>`d95c46a` | Escenarios de carga y generar gráficas (Issue #161) | @GemmaMartinez | @laloz60 | @laloz60 | 19/11/2026 | 21/11/2026 | [PR_179.pdf](pdf/PR_179.pdf) | [pr_179.png](screenshot/pr_179.png) |
| #180/#181<br>`2618a8f`<br>`2535c46` | docs: actualiza README con arranque, CI/CD, carga y FinOps (Issue #165); agrega `.github` en README | @GemmaMartinez | @laloz60 | @laloz60 | 21/11/2026 | 21/11/2026 | [PR_180.pdf](pdf/PR_180.pdf)<br>[PR_181.pdf](pdf/PR_181.pdf) | [pr_180.png](screenshot/pr_180.png)<br>[pr_181.png](screenshot/pr_181.png) |
| #174<br>`a6b168d` | feat(finops): estimación del costo por inferencia por escenario y réplicas | @smuciton | @GemmaMartinez | @GemmaMartinez | 19/11/2026 | 22/11/2026 | [PR_174.pdf](pdf/PR_174.pdf) | [pr_174.png](screenshot/pr_174.png) |
| #178<br>`caaa309` | docs(carga): evidencias de pruebas de carga, gráficas y capturas | @smuciton | @GemmaMartinez | @GemmaMartinez | 19/11/2026 | 22/11/2026 | [PR_178.pdf](pdf/PR_178.pdf) | [pr_178.png](screenshot/pr_178.png) |
| #183 | docs(finops): incluir costo por LCU del ALB en el escalado horizontal | @smuciton | @GemmaMartinez | @GemmaMartinez | 22/11/2026 | 22/11/2026 | [PR_183.pdf](pdf/PR_183.pdf) | [pr_183.png](screenshot/pr_183.png) |

**Nota:** El PR en donde se sube esta información se hará visible hasta la siguiente semana.

## Estructura de evidencias

Los PDF individuales se encuentran en `pdf/` y las capturas de cada PR en `screenshot/`.