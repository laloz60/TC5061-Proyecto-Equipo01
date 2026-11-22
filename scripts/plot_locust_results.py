from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

escenarios = ["A", "B", "C"]
patron_nombre = "{metrica}_{escenario}.png"

base_dir = Path("reports/load")
out_dir = base_dir / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

resumen = []

for escenario in escenarios:
  df = pd.read_csv(base_dir / f"{escenario}_stats_history.csv")
  if "Name" in df.columns:
    df = df[df["Name"] == "Aggregated"].copy()

  df["time"] = pd.to_datetime(df["Timestamp"], unit="s")

  # Latencia - p50, p95, p99
  plt.figure(figsize=(10, 5))
  for col, label in [("50%", "p50"), ("95%", "p95"), ("99%", "p99")]:
    if col in df.columns:
      plt.plot(df["time"], df[col], label=label)
  plt.title(f"Latencia por percentil - {escenario}")
  plt.xlabel("Tiempo")
  plt.ylabel("Latencia (ms)")
  plt.legend()
  plt.tight_layout()
  plt.savefig(
      out_dir / patron_nombre.format(metrica="latencia", escenario=escenario),
      dpi=150,)
  plt.close()

  # Peticiones por segundo
  plt.figure(figsize=(10, 5))
  plt.plot(df["time"], df["Requests/s"], label="RPS")
  plt.title(f"Peticiones por segundo - {escenario}")
  plt.xlabel("Tiempo")
  plt.ylabel("Peticiones/s")
  plt.legend()
  plt.tight_layout()
  plt.savefig(
      out_dir / patron_nombre.format(metrica="rps", escenario=escenario),
      dpi=150,)
  plt.close()

  # Tasa de error
  rps = df["Requests/s"].replace(0, pd.NA)
  error_rate = ((df["Failures/s"] / rps) * 100).fillna(0)

  plt.figure(figsize=(10, 5))
  plt.plot(df["time"], error_rate, label="% Tasa de error")
  plt.title(f"Tasa de error - {escenario}")
  plt.xlabel("Tiempo")
  plt.ylabel("% Tasa de error")
  plt.legend()
  plt.tight_layout()
  plt.savefig(
      out_dir / patron_nombre.format(metrica="error", escenario=escenario),
      dpi=150,)
  plt.close()

  # Datos finales de tabla comparativa
  stats = pd.read_csv(base_dir / f"{escenario}_stats.csv")
  agg = stats[stats["Name"] == "Aggregated"].iloc[0]
  total_req = agg["Request Count"]
  total_fail = agg["Failure Count"]

  resumen.append({
      "Escenario": escenario,
      "Peticiones": int(total_req),
      "Fallos": int(total_fail),
      "Error (%)": round((total_fail / total_req) * 100, 2) if total_req else 0,
      "RPS prom": round(agg["Requests/s"], 2),
      "p50 (ms)": agg["50%"],
      "p95 (ms)": agg["95%"],
      "p99 (ms)": agg["99%"],})

tabla = pd.DataFrame(resumen)
tabla.to_csv(base_dir / "comparativa_escenarios.csv", index=False)
print(tabla.to_string(index=False))