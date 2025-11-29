# Results

Esta carpeta contiene los resultados del proyecto.

## Contenido

Después de ejecutar el pipeline, aquí se guardarán:

- `metrics.json` - Métricas de evaluación de modelos
- `confusion_matrix_rf.png` - Matriz de confusión de Random Forest
- `confusion_matrix_svm.png` - Matriz de confusión de SVM
- Otras visualizaciones y reportes

## Estructura

```
results/
├── metrics.json
├── figures/
│   ├── exploratory/
│   ├── preprocessing/
│   └── evaluation/
└── models/
    ├── random_forest.pkl
    └── svm.pkl
```
