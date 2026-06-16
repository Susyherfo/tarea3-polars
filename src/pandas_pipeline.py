"""
pandas_pipeline.py
──────────────────
Pipeline equivalente al de Polars, implementado en Pandas.
Se usa exclusivamente para el benchmark de la Parte 6.
"""

import pandas as pd
import numpy as np
import time
import tracemalloc

NYC_LON = (-74.05, -73.75)
NYC_LAT = (40.63, 40.85)
DURACION_MIN = 60
DURACION_MAX = 7200
PASAJEROS_MIN = 1
PASAJEROS_MAX = 6

VENDOR_INFO_PD = pd.DataFrame({
    "vendor_id":   [1, 2],
    "vendor_name": ["Creative Mobile Technologies", "VeriFone Inc."],
    "vendor_tier": ["standard", "premium"],
})


def _medir(fn):
    """Ejecuta fn() y retorna (resultado, tiempo_s, mem_peak_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    resultado = fn()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return resultado, round(t1 - t0, 4), round(peak / (1024 ** 2), 2)


def ejecutar_pipeline_pandas(data_path: str, verbose: bool = True) -> dict:
    """
    Ejecuta el mismo pipeline que polars_pipeline.py pero en Pandas.
    Mide tiempo y memoria de cada etapa.

    Returns:
        dict con resultados y tiempos de cada etapa
    """
    tiempos = {}
    memorias = {}

    # ── 1. Lectura ────────────────────────────────────────────────────────────
    df, t, m = _medir(lambda: pd.read_csv(data_path))
    tiempos["lectura"] = t
    memorias["lectura"] = m
    if verbose:
        print(f"[Pandas] Lectura:       {t:.4f}s  |  {m:.1f} MB  |  {len(df):,} filas")

    # ── 2. Filtrado ───────────────────────────────────────────────────────────
    def _filtrar():
        mask = (
            (df["trip_duration"] > DURACION_MIN)
            & (df["trip_duration"] <= DURACION_MAX)
            & (df["passenger_count"] >= PASAJEROS_MIN)
            & (df["passenger_count"] <= PASAJEROS_MAX)
            & df["pickup_longitude"].between(*NYC_LON)
            & df["pickup_latitude"].between(*NYC_LAT)
            & df["dropoff_longitude"].between(*NYC_LON)
            & df["dropoff_latitude"].between(*NYC_LAT)
        )
        return df[mask].copy()
    df_f, t, m = _medir(_filtrar)
    tiempos["filtrado"] = t
    memorias["filtrado"] = m
    if verbose:
        print(f"[Pandas] Filtrado:      {t:.4f}s  |  {m:.1f} MB  |  {len(df_f):,} filas")

    # ── 3. Agregación ─────────────────────────────────────────────────────────
    def _agregar():
        return df_f.groupby("vendor_id")["trip_duration"].agg(["mean", "std", "count"])
    _, t, m = _medir(_agregar)
    tiempos["agregacion"] = t
    memorias["agregacion"] = m
    if verbose:
        print(f"[Pandas] Agregación:    {t:.4f}s  |  {m:.1f} MB")

    # ── 4. Join (merge) ───────────────────────────────────────────────────────
    def _join():
        return df_f.merge(VENDOR_INFO_PD, on="vendor_id", how="left")
    df_j, t, m = _medir(_join)
    tiempos["join"] = t
    memorias["join"] = m
    if verbose:
        print(f"[Pandas] Join:          {t:.4f}s  |  {m:.1f} MB")

    # ── 5. Feature Engineering ────────────────────────────────────────────────
    def _feat_eng():
        df_out = df_j.copy()
        df_out["pickup_dt"]      = pd.to_datetime(df_out["pickup_datetime"])
        df_out["pickup_hour"]    = df_out["pickup_dt"].dt.hour
        df_out["pickup_weekday"] = df_out["pickup_dt"].dt.dayofweek
        df_out["pickup_month"]   = df_out["pickup_dt"].dt.month
        df_out["es_hora_punta"]  = (
            df_out["pickup_hour"].between(7, 9)
            | df_out["pickup_hour"].between(17, 19)
        ).astype(int)
        df_out["es_fin_semana"]  = (df_out["pickup_weekday"] >= 5).astype(int)

        # Haversine (numpy)
        R    = 6371
        lat1 = np.radians(df_out["pickup_latitude"].values)
        lat2 = np.radians(df_out["dropoff_latitude"].values)
        dlat = lat2 - lat1
        dlon = np.radians(
            df_out["dropoff_longitude"].values - df_out["pickup_longitude"].values
        )
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        df_out["distancia_km"]      = 2 * R * np.arcsin(np.sqrt(a))
        df_out["log_trip_duration"] = np.log1p(df_out["trip_duration"])
        df_out["flag_enc"]          = (df_out["store_and_fwd_flag"] == "Y").astype(int)
        return df_out

    df_fe, t, m = _medir(_feat_eng)
    tiempos["feature_engineering"] = t
    memorias["feature_engineering"] = m
    if verbose:
        print(f"[Pandas] Feature Eng:   {t:.4f}s  |  {m:.1f} MB")

    tiempos["total"] = round(sum(tiempos.values()), 4)
    if verbose:
        print(f"\n[Pandas] TOTAL:         {tiempos['total']:.4f}s")

    return {
        "df": df_fe,
        "tiempos": tiempos,
        "memorias": memorias,
    }


if __name__ == "__main__":
    import sys
    import os
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/train.csv"

    # Crear carpetas de salida si no existen
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    resultado = ejecutar_pipeline_pandas(path)
    print(f"\nDataset final: {len(resultado['df']):,} filas × {resultado['df'].shape[1]} columnas")
