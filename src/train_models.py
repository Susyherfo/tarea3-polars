"""
train_models.py
───────────────
Entrenamiento y evaluación de modelos de Machine Learning.
Modelos: Ridge Regression · Random Forest · XGBoost
Target: log1p(trip_duration)  →  se evalúa también en escala original (segundos)
"""

import polars as pl
import numpy as np
import time
import os

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

from feature_engineering import FEATURE_COLS, TARGET_LOG, TARGET_ORIG


# ── Configuración de modelos ─────────────────────────────────────────────────
MODELOS_CONFIG = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    ),
}


def evaluar_modelo(nombre: str, y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """
    Calcula métricas en escala log y en escala original.

    Returns:
        dict con RMSE_log, MAE_log, R2, RMSE_seg, MAE_seg
    """
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log  = mean_absolute_error(y_true_log, y_pred_log)
    r2       = r2_score(y_true_log, y_pred_log)

    y_true_s = np.expm1(y_true_log)
    y_pred_s = np.expm1(np.clip(y_pred_log, 0, None))
    rmse_s   = np.sqrt(mean_squared_error(y_true_s, y_pred_s))
    mae_s    = mean_absolute_error(y_true_s, y_pred_s)

    print(f"\n── {nombre} ──")
    print(f"  RMSE (log):  {rmse_log:.4f}  |  MAE (log):  {mae_log:.4f}  |  R²: {r2:.4f}")
    print(f"  RMSE (seg):  {rmse_s:.1f}    |  MAE (seg):  {mae_s:.1f}")

    return {
        "modelo":    nombre,
        "RMSE_log":  round(rmse_log, 4),
        "MAE_log":   round(mae_log, 4),
        "R2":        round(r2, 4),
        "RMSE_seg":  round(rmse_s, 1),
        "MAE_seg":   round(mae_s, 1),
    }


def preparar_datos(processed_path: str, test_size: float = 0.20, seed: int = 42):
    """
    Carga el dataset procesado y genera los splits de train/test.

    Returns:
        X_train, X_test, y_train, y_test
    """
    t0 = time.time()
    df = pl.scan_csv(processed_path).collect()
    t_carga = time.time() - t0
    print(f"Dataset cargado: {df.shape[0]:,} filas  |  {t_carga:.3f}s")

    t0 = time.time()
    X = df.select(FEATURE_COLS).to_numpy()
    y = df[TARGET_LOG].to_numpy()
    t_prep = time.time() - t0
    print(f"Conversión a numpy: {t_prep:.3f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


def entrenar_todos(processed_path: str, results_dir: str = "results") -> pl.DataFrame:
    """
    Entrena los tres modelos y guarda los resultados.

    Returns:
        DataFrame de Polars con métricas y tiempos de todos los modelos
    """
    os.makedirs(results_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = preparar_datos(processed_path)

    resultados = []
    tiempos    = {}

    for nombre, modelo in MODELOS_CONFIG.items():
        print(f"\n{'─'*50}")
        print(f"Entrenando: {nombre}")

        kwargs = {}
        if nombre == "XGBoost":
            kwargs = {"eval_set": [(X_test, y_test)], "verbose": 50}

        t0 = time.time()
        modelo.fit(X_train, y_train, **kwargs)
        t_train = round(time.time() - t0, 3)

        t0 = time.time()
        y_pred = modelo.predict(X_test)
        t_pred = round(time.time() - t0, 4)

        tiempos[nombre] = {"entrenamiento": t_train, "prediccion": t_pred}
        print(f"  Tiempo entrenamiento: {t_train}s  |  Predicción: {t_pred}s")

        metricas = evaluar_modelo(nombre, y_test, y_pred)
        metricas["t_entrenamiento_s"] = t_train
        metricas["t_prediccion_s"]    = t_pred
        resultados.append(metricas)

    df_resultados = pl.DataFrame(resultados)

    # Guardar tabla de resultados
    df_resultados.write_csv(f"{results_dir}/metricas_modelos.csv")
    print(f"\n✓ Métricas guardadas en {results_dir}/metricas_modelos.csv")

    print("\n" + "=" * 60)
    print("TABLA FINAL DE RESULTADOS")
    print("=" * 60)
    print(df_resultados.to_pandas().to_string(index=False))

    return df_resultados


if __name__ == "__main__":
    import sys
    processed = sys.argv[1] if len(sys.argv) > 1 else "data/processed/nyc_taxi_processed.csv"
    entrenar_todos(processed)
