"""
Módulo para carga y gestión de datos del dataset de radiografías.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DataLoader:
    """
    Carga y gestiona el dataset de radiografías de tórax.
    
    Estructura esperada:
    data_root/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
    """
    
    CLASSES = ['NORMAL', 'PNEUMONIA']
    SPLITS = ['train', 'val', 'test']
    
    def __init__(self, data_root: str):
        """
        Inicializa el cargador de datos.
        
        Args:
            data_root: Ruta raíz del dataset
        """
        self.data_root = Path(data_root)
        self.image_paths = self._scan_dataset()
    
    def _scan_dataset(self) -> Dict[str, Dict[str, List[Path]]]:
        """Escanea el dataset y obtiene las rutas de todas las imágenes."""
        image_paths = {}
        
        for split in self.SPLITS:
            split_path = self.data_root / split
            if not split_path.exists():
                print(f"Advertencia: No se encontró {split_path}")
                continue
                
            image_paths[split] = {}
            for class_name in self.CLASSES:
                class_path = split_path / class_name
                if class_path.exists():
                    paths = list(class_path.glob('*.jpeg')) + \
                            list(class_path.glob('*.jpg')) + \
                            list(class_path.glob('*.png'))
                    image_paths[split][class_name] = paths
                else:
                    image_paths[split][class_name] = []
                    
        return image_paths
    
    def get_image_paths(self) -> Dict[str, Dict[str, List[Path]]]:
        """Retorna las rutas de imágenes organizadas por split y clase."""
        return self.image_paths
    
    def get_class_counts(self) -> Dict[str, Dict[str, int]]:
        """Retorna el conteo de imágenes por split y clase."""
        counts = {}
        for split, classes in self.image_paths.items():
            counts[split] = {cls: len(paths) for cls, paths in classes.items()}
        return counts
    
    def load_image(self, path: Path, grayscale: bool = True) -> Optional[np.ndarray]:
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
    
    def load_split(self, split: str, 
                   max_per_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga todas las imágenes de un split.
        
        Args:
            split: 'train', 'val', o 'test'
            max_per_class: Límite de imágenes por clase (para pruebas rápidas)
            
        Returns:
            Tuple[imágenes, etiquetas]
        """
        images, labels = [], []
        
        for class_idx, class_name in enumerate(self.CLASSES):
            paths = self.image_paths[split][class_name]
            if max_per_class:
                paths = paths[:max_per_class]
            
            for path in paths:
                img = self.load_image(path)
                if img is not None:
                    images.append(img)
                    labels.append(class_idx)
        
        return images, np.array(labels)
    
    def get_sample_images(self, split: str = 'train', 
                         n_per_class: int = 5) -> Dict[str, List[np.ndarray]]:
        """
        Obtiene imágenes de muestra de cada clase.
        
        Args:
            split: Split a usar
            n_per_class: Número de imágenes por clase
            
        Returns:
            Dict con imágenes por clase
        """
        samples = {}
        for class_name in self.CLASSES:
            paths = self.image_paths[split][class_name][:n_per_class]
            samples[class_name] = [self.load_image(p) for p in paths]
        return samples
    
    def analyze_dimensions(self, n_samples: int = 100) -> Dict:
        """
        Analiza las dimensiones de las imágenes del dataset.
        
        Args:
            n_samples: Número de muestras por clase
            
        Returns:
            Dict con estadísticas de dimensiones
        """
        heights, widths = [], []
        
        for split in self.SPLITS:
            for class_name in self.CLASSES:
                paths = self.image_paths.get(split, {}).get(class_name, [])
                for path in paths[:n_samples]:
                    img = self.load_image(path)
                    if img is not None:
                        heights.append(img.shape[0])
                        widths.append(img.shape[1])
        
        return {
            'mean_height': np.mean(heights),
            'mean_width': np.mean(widths),
            'std_height': np.std(heights),
            'std_width': np.std(widths),
            'min_height': np.min(heights),
            'max_height': np.max(heights),
            'min_width': np.min(widths),
            'max_width': np.max(widths)
        }
