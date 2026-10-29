# Laboratorio #9 — Series de tiempo y forecasting

**CC3092 · Deep Learning y Sistemas Inteligentes** — Universidad del Valle de Guatemala
**Nicolás Concuá y Esteban**

Pronóstico *day-ahead* (H = 24 h) del consumo eléctrico horario de 100 clientes de Portugal
(ElectricityLoadDiagrams20112014, UCI), evaluado con un único `backtest(modelo)` de origen móvil diario
(92 orígenes, oct–dic 2014).

## Estructura

```
Lab9-DL/
├── notebook/
│   └── lab9.ipynb               # notebook completo y ejecutado
├── reports/                     # figuras (.png), tablas (.csv) y pronósticos de prueba
│   └── pronosticos_prueba.parquet   # unique_id, ds, modelo, y_hat (+ p10/p50/p90)
└── data/                        # datos crudos, cachés y pesos (no se versiona)
```

## Cómo reproducir

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install torch pandas numpy matplotlib statsforecast holidays pyarrow jupyter
cd notebook
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 lab9.ipynb
```

Los datos se descargan automáticamente. Los pesos entrenados se guardan en `data/modelos/` y se reutilizan al
re-ejecutar; `LAB9_RETRAIN=1` fuerza a reentrenar y `LAB9_QUICK=1` corre una prueba rápida. El notebook detecta
CUDA, MPS (Apple Silicon) o CPU. Hardware usado: Apple M4 Pro (MPS).
