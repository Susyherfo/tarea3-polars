"""
polars_pipeline.py
──────────────────
Pipeline completo de procesamiento de datos usando Polars.
Mide el tiempo de cada etapa para el benchmark.
"""

import polars as pl
import numpy as np
import time
import tracemalloc

from preprocessing import (
    NYC_LON, NYC_LAT, DURACION_MIN, DURACION_MAX,
    PASAJEROS_MIN, PASAJEROS_MAX,
)
from feature_engineering import haversine_expr, VENDOR_INFO


def _medir(fn):
    """Ejecuta fn() y retorna (resultado, tiempo_s, mem_peak_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    resultado = fn()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return resultado, round(t1 - t0, 4), round(peak / (1024 ** 2), 2)


def ejecutar_pipeline_polars(data_path: str, verbose: bool = True) -> dict:
    """
    Ejecuta el pipeline de Polars etapa por etapa midiendo tiempos y memoria.

    Returns:
        dict con resultados y tiempos de cada etapa
    """
    tiempos = {}
    memorias = {}

    # ── 1. Lectura ────────────────────────────────────────────────────────────
    df, t, m = _medir(lambda: pl.read_csv(data_path))
    tiempos["lectura"] = t
    memorias["lectura"] = m
    if verbose:
        print(f"[Polars] Lectura:       {t:.4f}s  |  {m:.1f} MB  |  {df.shape[0]:,} filas")

    # ── 2. Filtrado ───────────────────────────────────────────────────────────
    def _filtrar():
        return df.filter(
            (pl.col("trip_duration") > DURACION_MIN)
            & (pl.col("trip_duration") <= DURACION_MAX)
            & (pl.col("passenger_count") >= PASAJEROS_MIN)
            & (pl.col("passenger_count") <= PASAJEROS_MAX)
            & pl.col("pickup_longitude").is_between(*NYC_LON)
            & pl.col("pickup_latitude").is_between(*NYC_LAT)
            & pl.col("dropoff_longitude").is_between(*NYC_LON)
            & pl.col("dropoff_latitude").is_between(*NYC_LAT)
        )
    df_f, t, m = _medir(_filtrar)
    tiempos["filtrado"] = t
    memorias["filtrado"] = m
    if verbose:
        print(f"[Polars] Filtrado:      {t:.4f}s  |  {m:.1f} MB  |  {df_f.shape[0]:,} filas")

    # ── 3. Agregación ─────────────────────────────────────────────────────────
    def _agregar():
        return df_f.group_by("vendor_id").agg([
            pl.col("trip_duration").mean().alias("dur_media"),
            pl.col("trip_duration").std().alias("dur_std"),
            pl.len().alias("n_viajes"),
        ])
    _, t, m = _medir(_agregar)
    tiempos["agregacion"] = t
    memorias["agregacion"] = m
    if verbose:
        print(f"[Polars] Agregación:    {t:.4f}s  |  {m:.1f} MB")

    # ── 4. Join ───────────────────────────────────────────────────────────────
    def _join():
        return df_f.join(VENDOR_INFO, on="vendor_id", how="left")
    df_j, t, m = _medir(_join)
    tiempos["join"] = t
    memorias["join"] = m
    if verbose:
        print(f"[Polars] Join:          {t:.4f}s  |  {m:.1f} MB")

    # ── 5. Feature Engineering ────────────────────────────────────────────────
    def _feat_eng():
        return (
            df_j
            .with_columns([
                pl.col("pickup_datetime")
                  .str.to_datetime("%Y-%m-%d %H:%M:%S").alias("pickup_dt"),
            ])
            .with_columns([
                pl.col("pickup_dt").dt.hour().alias("pickup_hour"),
                pl.col("pickup_dt").dt.weekday().alias("pickup_weekday"),
                pl.col("pickup_dt").dt.month().alias("pickup_month"),
                (
                    pl.col("pickup_dt").dt.hour().is_between(7, 9)
                    | pl.col("pickup_dt").dt.hour().is_between(17, 19)
                ).cast(pl.Int8).alias("es_hora_punta"),
                (pl.col("pickup_dt").dt.weekday() >= 5)
                 .cast(pl.Int8).alias("es_fin_semana"),
                haversine_expr(
                    pl.col("pickup_latitude"), pl.col("pickup_longitude"),
                    pl.col("dropoff_latitude"), pl.col("dropoff_longitude"),
                ).alias("distancia_km"),
                pl.col("trip_duration").log1p().alias("log_trip_duration"),
                (pl.col("store_and_fwd_flag") == "Y")
                  .cast(pl.Int8).alias("flag_enc"),
            ])
        )
    df_fe, t, m = _medir(_feat_eng)
    tiempos["feature_engineering"] = t
    memorias["feature_engineering"] = m
    if verbose:
        print(f"[Polars] Feature Eng:   {t:.4f}s  |  {m:.1f} MB")

    tiempos["total"] = round(sum(tiempos.values()), 4)
    if verbose:
        print(f"\n[Polars] TOTAL:         {tiempos['total']:.4f}s")

    return {
        "df": df_fe,
        "tiempos": tiempos,
        "memorias": memorias,
    }


def pipeline_lazy(data_path: str) -> tuple[pl.DataFrame, float]:
    """
    Pipeline usando LazyFrame (scan_csv + collect) para el experimento 7.2.

    Returns:
        (DataFrame resultado, tiempo en segundos)
    """
    t0 = time.perf_counter()
    result = (
        pl.scan_csv(data_path)
        .filter(
            (pl.col("trip_duration") > DURACION_MIN)
            & (pl.col("trip_duration") <= DURACION_MAX)
            & pl.col("pickup_longitude").is_between(*NYC_LON)
            & pl.col("pickup_latitude").is_between(*NYC_LAT)
        )
        .with_columns([
            pl.col("pickup_datetime")
              .str.to_datetime("%Y-%m-%d %H:%M:%S").alias("pickup_dt"),
        ])
        .with_columns([
            pl.col("pickup_dt").dt.hour().alias("pickup_hour"),
        ])
        .group_by("pickup_hour")
        .agg(pl.col("trip_duration").mean())
        .collect()
    )
    return result, round(time.perf_counter() - t0, 4)


def pipeline_eager(data_path: str) -> tuple[pl.DataFrame, float]:
    """
    Pipeline usando read_csv (Eager) para el experimento 7.2.

    Returns:
        (DataFrame resultado, tiempo en segundos)
    """
    t0 = time.perf_counter()
    df = pl.read_csv(data_path)
    result = (
        df
        .filter(
            (pl.col("trip_duration") > DURACION_MIN)
            & (pl.col("trip_duration") <= DURACION_MAX)
            & pl.col("pickup_longitude").is_between(*NYC_LON)
            & pl.col("pickup_latitude").is_between(*NYC_LAT)
        )
        .with_columns([
            pl.col("pickup_datetime")
              .str.to_datetime("%Y-%m-%d %H:%M:%S").alias("pickup_dt"),
        ])
        .with_columns([
            pl.col("pickup_dt").dt.hour().alias("pickup_hour"),
        ])
        .group_by("pickup_hour")
        .agg(pl.col("trip_duration").mean())
    )
    return result, round(time.perf_counter() - t0, 4)


if __name__ == "__main__":
    import sys
    import os
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/train.csv"

    # Crear carpetas de salida si no existen
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    resultado = ejecutar_pipeline_polars(path)
    print(f"\nDataset final: {resultado['df'].shape[0]:,} filas × {resultado['df'].shape[1]} columnas")
