"""
preprocessing.py
────────────────
Carga, limpieza y filtrado del dataset NYC Taxi Trip Duration.
Todas las operaciones usan Polars (LazyFrame cuando es posible).
"""

import polars as pl
import numpy as np

# ── Constantes ───────────────────────────────────────────────────────────────
NYC_LON = (-74.05, -73.75)
NYC_LAT = (40.63, 40.85)
DURACION_MIN = 60       # segundos
DURACION_MAX = 7200     # segundos (2 horas)
PASAJEROS_MIN = 1
PASAJEROS_MAX = 6


def cargar_dataset(path: str, lazy: bool = True) -> pl.DataFrame:
    """
    Carga el CSV usando LazyFrame por defecto.

    Args:
        path: Ruta al archivo train.csv
        lazy: Si True usa scan_csv (Lazy), si False usa read_csv (Eager)

    Returns:
        DataFrame de Polars materializado
    """
    if lazy:
        return pl.scan_csv(path).collect()
    return pl.read_csv(path)


def filtrar_registros(df: pl.DataFrame) -> pl.DataFrame:
    """
    Elimina registros inválidos:
      - Duración fuera del rango [60s, 7200s]
      - Pasajeros fuera del rango [1, 6]
      - Coordenadas fuera del área geográfica de NYC
    """
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


def manejar_nulos(df: pl.DataFrame) -> pl.DataFrame:
    """
    Imputa valores faltantes:
      - store_and_fwd_flag → 'N'
      - passenger_count    → mediana
    """
    mediana_pasajeros = df["passenger_count"].median()
    return df.with_columns([
        pl.col("store_and_fwd_flag").fill_null("N"),
        pl.col("passenger_count").fill_null(mediana_pasajeros),
    ])


def resumen_nulos(df: pl.DataFrame) -> pl.DataFrame:
    """Retorna una tabla con conteo y porcentaje de nulos por columna."""
    nulos = df.select([
        pl.col(c).null_count().alias(c) for c in df.columns
    ]).unpivot(variable_name="columna", value_name="nulos")
    return nulos.with_columns(
        (pl.col("nulos") / df.shape[0] * 100).round(2).alias("pct_nulos")
    )


def estadisticas_descriptivas(df: pl.DataFrame) -> pl.DataFrame:
    """Estadísticas descriptivas de columnas numéricas."""
    cols = [
        "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "trip_duration",
    ]
    return df.select(cols).describe()


def pipeline_preprocesamiento(path: str) -> pl.DataFrame:
    """
    Ejecuta el pipeline completo de preprocesamiento:
    carga → filtrado → manejo de nulos.

    Returns:
        DataFrame limpio listo para feature engineering
    """
    df = cargar_dataset(path, lazy=True)
    df = filtrar_registros(df)
    df = manejar_nulos(df)
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/train.csv"
    df = pipeline_preprocesamiento(path)
    print(f"Registros tras preprocesamiento: {df.shape[0]:,}")
    print(resumen_nulos(df))
