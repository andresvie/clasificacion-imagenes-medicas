# Clasificación de Neumonía en Radiografías de Tórax

## Universidad Nacional de Colombia
**Visión por Computador 3009228** - Semestre 2025-02  
Facultad de Minas - Departamento de Ciencias de la Computación y de la Decisión

---

## Descripción

Este proyecto implementa un pipeline de reconocimiento de patrones para la clasificación de neumonía en radiografías de tórax, utilizando el dataset de Kaggle [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Estructura del Proyecto

```
clasificacion-imagenes-medicas/
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
cd clasificacion-imagenes-medicas

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Dataset

Descargar el dataset de Kaggle y colocarlo en `data/chest-xray-pneumonia/`:

```bash
cd data/
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

La estructura esperada del dataset es:
```
data/chest-xray-pneumonia/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Uso

### Ejecutar el pipeline completo

```bash
python src/classification_pipeline.py --data_root ./data/chest-xray-pneumonia/chest_xray
```

Opciones disponibles:
- `--data_root`: Ruta al dataset (default: `./data/chest-xray-pneumonia/chest_xray`)
- `--results_dir`: Directorio para guardar resultados (default: `./results`)
- `--max_samples`: Máximo de muestras por clase (default: 0 = sin límite)

### Ejecutar notebooks

```bash
jupyter notebook notebooks/
# O usar JupyterLab
jupyter lab
```

### Ejecutar tests

```bash
# Ejecutar todos los tests
pytest tests/

# Ejecutar con cobertura
pytest --cov=src --cov-report=html tests/
```

## Módulos del Proyecto

El código fuente está organizado en los siguientes módulos en `src/`:

| Módulo | Descripción |
|--------|-------------|
| `data_loader.py` | Carga y gestión del dataset de radiografías |
| `preprocessing.py` | Preprocesamiento de imágenes (redimensionado, CLAHE, normalización) |
| `feature_extraction.py` | Extracción de características (intensidad, textura, forma) |
| `texture_descriptors.py` | Descriptores de textura (GLCM, LBP, Gabor, estadísticas de primer orden) |
| `shape_descriptors.py` | Descriptores de forma (HOG, momentos de Hu, contornos, Fourier) |
| `classification_pipeline.py` | Pipeline completo de clasificación |
| `utils.py` | Utilidades (carga/guardado de imágenes, visualizaciones) |

## Notebooks

| Notebook | Descripción |
|----------|-------------|
| `01_exploratory_and_preprocessing.ipynb` | Análisis exploratorio de datos y pipeline de preprocesamiento |
| `02_feature_extraction.ipynb` | Extracción y análisis de características (textura y forma) |

## Características Extraídas

El pipeline extrae múltiples tipos de características:

### Características de Intensidad
- Estadísticas de primer orden (media, varianza, asimetría, curtosis, etc.)

### Características de Textura
- **GLCM** (Gray-Level Co-occurrence Matrix): Contraste, homogeneidad, energía, correlación
- **LBP** (Local Binary Patterns): Patrones binarios locales
- **Filtros de Gabor**: Respuestas a diferentes frecuencias y orientaciones
- **Gradientes**: Magnitud y orientación de gradientes

### Características de Forma
- **HOG** (Histogram of Oriented Gradients)
- **Momentos de Hu**: 7 momentos invariantes
- **Descriptores de contorno**: Área, perímetro, circularidad, excentricidad
- **Descriptores de Fourier**: Representación espectral de formas

## Resultados

Los resultados se guardan en el directorio `results/`:
- **Figuras y visualizaciones**: Distribuciones, comparaciones, matrices de confusión
- **Métricas de evaluación**: Accuracy, AUC-ROC, precision, recall, F1-score (en `metrics.json`)
- **Descriptores extraídos**: Vectores de características guardados en formato NumPy
- **Análisis de preprocesamiento**: Efectos de CLAHE y otras transformaciones

## Modelos Implementados

- **Random Forest**: Clasificador basado en árboles de decisión
- **SVM** (Support Vector Machine): Clasificador con kernel RBF

## Contribuciones

Este trabajo fue desarrollado en equipo por:

### Parte 1
- Carlos Andrés Viera Mosquera (cviera@unal.edu.co)

### Parte 2
- Yenifer Tatiana Guavita Ospino (yguavita@unal.edu.co)
- Yojan Tamayo Montoya (ytamayom@unal.edu.co)

### Parte 3
- Lina María Montoya Zuluaga (limontoyaz@unal.edu.co)

## Enlaces Útiles

- [GitHub Pages del Proyecto](https://carlosviera.github.io/clasificacion-imagenes-medicas/)
- [Documentación de OpenCV](https://docs.opencv.org/)

## Licencia

Este proyecto es parte de un trabajo académico de la Universidad Nacional de Colombia.

---

**Universidad Nacional de Colombia**  
Visión por Computador 3009228 - Semestre 2025-02
