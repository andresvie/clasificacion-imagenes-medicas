# Clasificación de Neumonía en Radiografías de Tórax

## Universidad Nacional de Colombia
**Visión por Computador 3009228** - Semestre 2025-02  
Facultad de Minas - Departamento de Ciencias de la Computación y de la Decisión

---

## Descripción

Este proyecto implementa un pipeline de reconocimiento de patrones para la clasificación de neumonía en radiografías de tórax, utilizando el dataset de Kaggle [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Estructura del Proyecto

```
pneumonia_project/
├── data/                    # Datos y scripts de descarga
├── notebooks/               # Jupyter notebooks con análisis
├── results/                 # Resultados, figuras y métricas
├── src/                     # Código fuente del proyecto
├── tests/                   # Tests automatizados
├── README.md
├── requirements.txt
├── pytest.ini
├── _config.yml
└── index.html
```

## Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd pneumonia_project

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Dataset

Descargar el dataset de Kaggle y colocarlo en `data/chest_xray/`:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

## Uso

### Ejecutar el pipeline completo

```bash
python src/main.py
```

### Ejecutar notebooks

```bash
jupyter notebook notebooks/
```

### Ejecutar tests

```bash
pytest tests/
```

## Notebooks

| Notebook | Descripción |
|----------|-------------|
| `01_exploratory_analysis.ipynb` | Análisis exploratorio de datos |
| `02_preprocessing.ipynb` | Pipeline de preprocesamiento |
| `03_feature_extraction.ipynb` | Extracción de características |
| `04_classification.ipynb` | Entrenamiento y evaluación de modelos |

## Resultados

Los resultados se guardan en el directorio `results/`:
- Figuras y visualizaciones
- Métricas de evaluación
- Modelos entrenados

## Tests

```bash
# Ejecutar todos los tests
pytest

# Ejecutar con cobertura
pytest --cov=src tests/
```

## Autor

Universidad Nacional de Colombia  
Visión por Computador 3009228 - Semestre 2025-02
