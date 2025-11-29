"""
Funciones de utilidad para el proyecto.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt


def load_image(path: str, grayscale: bool = True) -> Optional[np.ndarray]:
    """
    Carga una imagen desde disco.
    
    Args:
        path: Ruta a la imagen
        grayscale: Si True, carga en escala de grises
        
    Returns:
        np.ndarray o None si hay error
    """
    if grayscale:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(image: np.ndarray, path: str) -> bool:
    """
    Guarda una imagen en disco.
    
    Args:
        image: Imagen a guardar
        path: Ruta de destino
        
    Returns:
        True si se guardó correctamente
    """
    try:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image)
        return True
    except Exception as e:
        print(f"Error guardando imagen: {e}")
        return False


def save_results(results: Dict[str, Any], path: str) -> None:
    """
    Guarda resultados en formato JSON.
    
    Args:
        results: Diccionario con resultados
        path: Ruta de destino
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convertir numpy arrays a listas
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)


def load_results(path: str) -> Dict[str, Any]:
    """
    Carga resultados desde JSON.
    
    Args:
        path: Ruta del archivo
        
    Returns:
        Diccionario con resultados
    """
    with open(path, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(cm: np.ndarray, 
                         classes: list,
                         normalize: bool = False,
                         title: str = 'Confusion Matrix',
                         cmap: str = 'Blues',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualiza una matriz de confusión.
    
    Args:
        cm: Matriz de confusión
        classes: Lista de nombres de clases
        normalize: Si normalizar valores
        title: Título del gráfico
        cmap: Mapa de colores
        save_path: Ruta para guardar la figura
        
    Returns:
        Figura de matplotlib
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Etiqueta Real',
           xlabel='Etiqueta Predicha')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(fpr: np.ndarray, 
                  tpr: np.ndarray, 
                  auc_score: float,
                  title: str = 'Curva ROC',
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualiza la curva ROC.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc_score: Área bajo la curva
        title: Título del gráfico
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def print_classification_report(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               classes: list) -> str:
    """
    Genera un reporte de clasificación formateado.
    
    Args:
        y_true: Etiquetas reales
        y_pred: Etiquetas predichas
        classes: Nombres de clases
        
    Returns:
        Reporte como string
    """
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, target_names=classes)
