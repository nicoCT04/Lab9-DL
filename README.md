# Laboratorio #9 — Series de tiempo y forecasting

**CC3092 · Deep Learning y Sistemas Inteligentes** — Universidad del Valle de Guatemala
**Nicolás Concuá y Esteban Cárcamo**

Pronóstico *day-ahead* (H = 24 h) del consumo eléctrico horario de 100 clientes de Portugal
(ElectricityLoadDiagrams20112014, UCI), evaluado con un único `backtest(modelo)` de origen móvil diario
(92 orígenes, oct–dic 2014).

## Resultados principales (prueba)

| Modelo | Familia | MASE | MAE (kWh) | MASE Navidad |
|---|---|---|---|---|
| iTransformer | Atención | **0.752** | 31.3 | 1.252 |
| Chronos-Bolt (zero-shot) | Fundacional | 0.770 | 31.7 | 1.443 |
| N-HiTS | Residuales y convolucionales | 0.806 | 33.9 | 1.647 |
| TCN | Residuales y convolucionales | 0.808 | 33.9 | 1.544 |
| MLP (lr = 3·10⁻⁴) | MLP y lineales | 0.811 | 33.4 | 1.338 |
| LightGBM | Árboles | 0.854 | 34.5 | 1.505 |
| Seasonal naive 24 h | Baseline | 1.034 | 44.1 | 2.168 |

La tabla completa (17 modelos, con tiempos, parámetros, MASE con H = 168, pinball y cobertura) está en
`reports/tabla_comparativa.csv` y en el informe.

## Estructura

```
Lab9-DL/
├── Informe_Lab9_DL.pdf          # informe (3 páginas)
├── notebook/
│   └── lab9.ipynb               # notebook completo y ejecutado
├── reports/                     # figuras (.png), tablas (.csv) y pronósticos de prueba
│   ├── tabla_comparativa.csv        # mejor configuración de cada modelo, ordenada por MASE
│   ├── tabla_iteraciones.csv        # iteraciones de hiperparámetros (solo validación)
│   └── pronosticos_prueba.parquet   # unique_id, ds, modelo, y_hat (+ p10/p50/p90)
├── requirements.txt             # versiones probadas
└── data/                        # datos crudos, cachés y pesos (no se versiona)
```

## Cómo reproducir

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd notebook
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 lab9.ipynb
```

Los datos se descargan automáticamente (y los pesos de Chronos-Bolt, de Hugging Face). Los pesos entrenados se
guardan en `data/modelos/` y se reutilizan al re-ejecutar; `LAB9_RETRAIN=1` fuerza a reentrenar y `LAB9_QUICK=1`
corre una prueba rápida. El notebook detecta CUDA, MPS (Apple Silicon) o CPU.

**Hardware.** Los modelos se entrenaron en una laptop con Intel Core i9-13980HX y NVIDIA RTX 4050 Laptop (6 GB);
el desarrollo inicial, en un Apple M4 Pro (MPS). La ejecución final del notebook se hizo solo en CPU con los pesos en
caché (`CUDA_VISIBLE_DEVICES=''`), porque la laptop se apagaba por temperatura con la GPU a plena carga: las métricas
no cambian, pero los tiempos de inferencia reportados son de CPU.
