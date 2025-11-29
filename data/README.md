# Data Directory

Este directorio contiene el dataset de radiografías de tórax.

## Descarga del Dataset

Descargar desde Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Opción 1: Usando Kaggle CLI

```bash
# Instalar kaggle CLI
pip install kaggle

# Configurar API key (ver https://www.kaggle.com/docs/api)
# Descargar dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Descomprimir
unzip chest-xray-pneumonia.zip
```

### Opción 2: Descarga Manual

1. Ir a https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Descargar el archivo ZIP
3. Descomprimir en este directorio

## Estructura Esperada

```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   │   ├── IM-0115-0001.jpeg
    │   │   └── ...
    │   └── PNEUMONIA/
    │       ├── person1_bacteria_1.jpeg
    │       └── ...
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## Estadísticas del Dataset

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |
