"""
feature_engineering.py
───────────────────────
Ingeniería de características usando Polars.
Incluye features temporales, geográficas y de contexto de tráfico.
"""

import polars as pl
import numpy as np


# ── Tabla auxiliar de proveedores ────────────────────────────────────────────
VENDOR_INFO = pl.DataFrame({
    "vendor_id":   [1, 2],
    "vendor_name": ["Creative Mobile Technologies", "VeriFone Inc."],
    "vendor_tier": ["standard", "premium"],
})


def haversine_expr(lat1, lon1, lat2, lon2):
    """
    Fórmula de Haversine implementada con expresiones nativas de Polars.
    Retorna distancia en kilómetros.
    """
    R = 6371.0
    dlat = (lat2 - lat1) * np.pi / 180
    dlon = (lon2 - lon1) * np.pi / 180
    a = (
        (dlat / 2).sin() ** 2
        + lat1.radians().cos() * lat2.radians().cos() * (dlon / 2).sin() ** 2
    )
    c = 2 * a.sqrt().arcsin()
    return c * R


def agregar_features_temporales(df: pl.DataFrame) -> pl.DataFrame:
    """
    Añade columnas derivadas de pickup_datetime:
      pickup_hour, pickup_weekday, pickup_month, pickup_day,
      es_hora_punta, es_fin_semana, es_nocturno
    """
    return (
        df
        .with_columns([
            pl.col("pickup_datetime")
              .str.to_datetime("%Y-%m-%d %H:%M:%S")
              .alias("pickup_dt"),
            pl.col("dropoff_datetime")
              .str.to_datetime("%Y-%m-%d %H:%M:%S")
              .alias("dropoff_dt"),
        ])
        .with_columns([
            pl.col("pickup_dt").dt.hour().alias("pickup_hour"),
            pl.col("pickup_dt").dt.weekday().alias("pickup_weekday"),
            pl.col("pickup_dt").dt.month().alias("pickup_month"),
            pl.col("pickup_dt").dt.day().alias("pickup_day"),
            (
                pl.col("pickup_dt").dt.hour().is_between(7, 9)
                | pl.col("pickup_dt").dt.hour().is_between(17, 19)
            ).cast(pl.Int8).alias("es_hora_punta"),
            (pl.col("pickup_dt").dt.weekday() >= 5)
             .cast(pl.Int8).alias("es_fin_semana"),
            (
                pl.col("pickup_dt").dt.hour().is_between(22, 23)
                | pl.col("pickup_dt").dt.hour().is_between(0, 5)
            ).cast(pl.Int8).alias("es_nocturno"),
        ])
    )


def agregar_features_geograficas(df: pl.DataFrame) -> pl.DataFrame:
    """
    Añade:
      distancia_km          - distancia Haversine entre pickup y dropoff
      delta_coord_manhattan - diferencia absoluta de coordenadas (proxy distancia)
      bearing_aprox         - dirección aproximada del viaje
    """
    return df.with_columns([
        haversine_expr(
            pl.col("pickup_latitude"),
            pl.col("pickup_longitude"),
            pl.col("dropoff_latitude"),
            pl.col("dropoff_longitude"),
        ).alias("distancia_km"),
        (
            (pl.col("pickup_longitude") - pl.col("dropoff_longitude")).abs()
            + (pl.col("pickup_latitude") - pl.col("dropoff_latitude")).abs()
        ).alias("delta_coord_manhattan"),
        (
            (pl.col("dropoff_longitude") - pl.col("pickup_longitude"))
            / (
                (pl.col("dropoff_latitude") - pl.col("pickup_latitude")).abs()
                + 1e-6
            )
        ).alias("bearing_aprox"),
    ])


def agregar_target_transformado(df: pl.DataFrame) -> pl.DataFrame:
    """Añade log1p(trip_duration) como variable objetivo normalizada."""
    return df.with_columns([
        pl.col("trip_duration").log1p().alias("log_trip_duration"),
        (pl.col("store_and_fwd_flag") == "Y")
          .cast(pl.Int8).alias("store_and_fwd_flag_enc"),
    ])


def agregar_velocidad_por_hora(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula velocidad media (km/h) por hora del día (group_by)
    y la une como feature de contexto de tráfico.
    """
    vel_hora = (
        df
        .with_columns([
            (pl.col("distancia_km") / (pl.col("trip_duration") / 3600))
              .alias("velocidad_kmh")
        ])
        .group_by("pickup_hour")
        .agg(pl.col("velocidad_kmh").mean().alias("vel_media_kmh"))
        .sort("pickup_hour")
    )
    return df.join(
        vel_hora.select(["pickup_hour", "vel_media_kmh"]),
        on="pickup_hour",
        how="left",
    )


def join_proveedores(df: pl.DataFrame) -> pl.DataFrame:
    """Une la tabla de proveedores por vendor_id."""
    return df.join(VENDOR_INFO, on="vendor_id", how="left")


def pipeline_feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pipeline completo de ingeniería de características.
    Aplica en orden: temporales → geográficas → target → join → velocidad.

    Returns:
        DataFrame con todas las features generadas
    """
    df = agregar_features_temporales(df)
    df = agregar_features_geograficas(df)
    df = agregar_target_transformado(df)
    df = agregar_velocidad_por_hora(df)
    df = join_proveedores(df)
    return df


# Columnas finales para ML
FEATURE_COLS = [
    "vendor_id", "passenger_count",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "store_and_fwd_flag_enc",
    "pickup_hour", "pickup_weekday", "pickup_month", "pickup_day",
    "es_hora_punta", "es_fin_semana", "es_nocturno",
    "distancia_km", "delta_coord_manhattan", "bearing_aprox",
    "vel_media_kmh",
]
TARGET_LOG  = "log_trip_duration"
TARGET_ORIG = "trip_duration"


if __name__ == "__main__":
    import sys
    import os
    from preprocessing import pipeline_preprocesamiento

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/train.csv"

    # Crear carpetas si no existen
    os.makedirs("data/processed", exist_ok=True)

    df_clean = pipeline_preprocesamiento(path)
    df_fe    = pipeline_feature_engineering(df_clean)
    print(f"Dataset con features: {df_fe.shape[0]:,} filas × {df_fe.shape[1]} columnas")
    print("Features generadas:", FEATURE_COLS)

    output_path = "data/processed/nyc_taxi_processed.csv"
    df_fe.select(FEATURE_COLS + [TARGET_LOG, TARGET_ORIG]).write_csv(output_path)
    print(f"✓ Guardado en {output_path}")
