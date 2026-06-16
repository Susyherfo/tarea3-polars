# Tarea 3: Polars — NYC Taxi Trip Duration

**Lead University · Curso: Ciencia de Datos · Profesor: Johansell Villalobos Cubillo**

---

## Descripción del problema

El objetivo es predecir la **duración de viajes de taxi en Nueva York** (en segundos) a partir de variables como la hora de recogida, la ubicación y el número de pasajeros. Además de construir el modelo predictivo, se realiza una **comparación sistemática de rendimiento entre Polars y Pandas** en cada etapa del pipeline.

---

## Dataset

| Atributo | Valor |
|---|---|
| Fuente | [Kaggle — NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) |
| Registros (raw) | 609 990 |
| Registros (tras filtrado) | 776 974 |
| Tamaño del archivo | 104 MB |
| Período | Enero – Junio 2016 |
| Variable objetivo | `trip_duration` (segundos) |

### Cómo obtener el dataset

1. Crear cuenta en [Kaggle](https://www.kaggle.com) y aceptar las reglas de la competencia.
2. Instalar la API de Kaggle: `pip install kaggle`
3. Colocar tu `kaggle.json` en `~/.kaggle/`
4. Ejecutar:

```bash
kaggle competitions download -c nyc-taxi-trip-duration
unzip nyc-taxi-trip-duration.zip -d data/raw/
```

---

## Requisitos de software

| Paquete | Versión |
|---|---|
| Python | 3.10+ |
| polars | 1.35.2 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| scikit-learn | 1.6.1 |
| xgboost | 3.2.0 |
| matplotlib | 3.9.0 |
| seaborn | 0.13.2 |
| psutil | 6.0.0 |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://gitlab.com/<usuario>/tarea3-polars.git
cd tarea3-polars

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Instrucciones de ejecución

### Opción A — Notebook completo (recomendado)

```bash
jupyter notebook notebooks/analysis.ipynb
```

El notebook ejecuta automáticamente todas las partes en orden: EDA → Feature Engineering → ML → Benchmark → Experimentos.

### Opción B — Scripts individuales

```bash
# 1. Preprocesamiento
python src/preprocessing.py data/raw/train.csv

# 2. Feature Engineering (genera data/processed/nyc_taxi_processed.csv)
python src/feature_engineering.py data/raw/train.csv

# 3. Pipeline Polars con tiempos
python src/polars_pipeline.py data/raw/train.csv

# 4. Pipeline Pandas con tiempos (benchmark)
python src/pandas_pipeline.py data/raw/train.csv

# 5. Entrenamiento de modelos ML
python src/train_models.py data/processed/nyc_taxi_processed.csv
```

---

## Estructura del repositorio

```
tarea3-polars/
│
├── data/
│   ├── raw/                   # train.csv (no incluido en el repo, ver instrucciones)
│   └── processed/             # nyc_taxi_processed.csv (generado por feature_engineering.py)
│
├── notebooks/
│   └── analysis.ipynb         # Notebook principal con todas las partes
│
├── src/
│   ├── preprocessing.py       # Carga, limpieza y filtrado del dataset
│   ├── feature_engineering.py # Creación de features (Polars)
│   ├── polars_pipeline.py     # Pipeline completo en Polars con medición de tiempos
│   ├── pandas_pipeline.py     # Pipeline equivalente en Pandas (benchmark)
│   └── train_models.py        # Entrenamiento de Ridge, Random Forest y XGBoost
│
├── figures/                   # Gráficas generadas por el notebook
├── results/                   # Tablas de métricas y tiempos (CSV)
├── report/
│   └── report.pdf             # Informe final en PDF
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Resumen de resultados

### Machine Learning

| Modelo | RMSE (log) | R² | RMSE (seg) | MAE (seg) | Tiempo entreno |
|---|---|---|---|---|---|
| Ridge Regression | 0.5002 | 0.5206 | 601.6 s | 318.8 s | 0.68 s |
| Random Forest | 0.3390 | 0.7798 | 299.2 s | 184.6 s | 773.32 s |
| **XGBoost** | **0.3267** | **0.7956** | **289.8 s** | **177.1 s** | **28.25 s** |

> **Mejor modelo: XGBoost** con R² = 0.7956 y error medio de ~3 minutos.  
> Las features más importantes fueron `distancia_km` y `delta_coord_manhattan`.

### Benchmark Polars vs Pandas

| Etapa | Polars | Pandas | Speedup |
|---|---|---|---|
| Lectura CSV | 0.907 s | 6.667 s | **7.3x** |
| Filtrado | 0.067 s | 0.217 s | 3.2x |
| Agregación | 0.080 s | 0.050 s | 0.6x |
| Join | 0.021 s | 0.140 s | 6.6x |
| Feature Engineering | 0.299 s | 1.156 s | 3.9x |
| **TOTAL** | **1.375 s** | **8.230 s** | **5.99x** |

### Experimento de Lazy Execution

| Modo | Tiempo promedio |
|---|---|
| Eager (`read_csv`) | 0.859 s |
| Lazy (`scan_csv + collect`) | 0.478 s — **1.80x más rápido** |

> Sistema: 2 núcleos CPU · 12.7 GB RAM · Dataset 104 MB

---

## Notas

- El notebook fue desarrollado y ejecutado en **Google Colab**.
- Polars versión **1.35.2** · Python **3.10**.
- El dataset de test de Kaggle no tiene `trip_duration` (es la variable a predecir para la competencia), por lo que se usó únicamente `train.csv` con split 80/20.
