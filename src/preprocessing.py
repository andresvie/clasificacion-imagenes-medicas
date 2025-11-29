"""
Pipeline de preprocesamiento para radiografías de tórax.

Incluye:
- Normalización de tamaño
- Ecualización CLAHE
- Segmentación de región de interés (opcional)
"""

import numpy as np
import cv2
from skimage import measure
from skimage.segmentation import clear_border


class XRayPreprocessor:
    """
    Pipeline de preprocesamiento para radiografías de tórax.
    
    Justificación de decisiones:
    
    1. Normalización de tamaño (224x224):
       - Estándar para CNNs pre-entrenadas (ImageNet)
       - Balance entre preservar detalles y eficiencia computacional
       - Permite usar transfer learning con VGG, ResNet, etc.
    
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization):
       - Superior a ecualización global para imágenes médicas
       - Mejora contraste local sin amplificar ruido excesivamente
       - El parámetro clipLimit controla la amplificación máxima
       - tileGridSize define las regiones para ecualización local
    
    3. Segmentación de pulmones (opcional):
       - Reduce ruido de estructuras no relevantes
       - Enfoca el análisis en la región de interés
    """
    
    def __init__(self, 
                 target_size: tuple = (224, 224),
                 apply_clahe: bool = True,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid: tuple = (8, 8),
                 segment_lungs: bool = False,
                 normalize: bool = True):
        """
        Inicializa el preprocesador.
        
        Args:
            target_size: Tamaño objetivo (altura, ancho)
            apply_clahe: Si aplicar ecualización CLAHE
            clahe_clip_limit: Límite de contraste para CLAHE (2.0-4.0 recomendado)
            clahe_tile_grid: Tamaño de la cuadrícula para CLAHE
            segment_lungs: Si aplicar segmentación de pulmones
            normalize: Si normalizar valores al rango [0, 1]
        """
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.segment_lungs = segment_lungs
        self.normalize = normalize
        
        if apply_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_tile_grid
            )
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Redimensiona la imagen al tamaño objetivo."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Aplica CLAHE para mejorar el contraste local."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return self.clahe.apply(image)
    
    def segment_lung_region(self, image: np.ndarray) -> tuple:
        """
        Segmenta la región pulmonar usando técnicas morfológicas.
        
        Returns:
            tuple: (imagen segmentada, máscara binaria)
        """
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()
        
        # Umbralización de Otsu
        _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = clear_border(binary)
        
        # Encontrar componentes conectados
        labels = measure.label(binary)
        regions = measure.regionprops(labels)
        regions = sorted(regions, key=lambda x: x.area, reverse=True)
        
        # Crear máscara con los dos pulmones
        mask = np.zeros_like(binary)
        for region in regions[:2]:
            if region.area > 1000:
                mask[labels == region.label] = 255
        
        mask = cv2.dilate(mask, kernel, iterations=2)
        segmented = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)
        
        return segmented, mask
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normaliza la imagen al rango [0, 1]."""
        return image.astype(np.float32) / 255.0
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica el pipeline completo de preprocesamiento.
        
        Args:
            image: Imagen de entrada en escala de grises
            
        Returns:
            np.ndarray: Imagen preprocesada
        """
        processed = self.resize(image)
        
        if self.apply_clahe:
            processed = self.apply_clahe_enhancement(processed)
        
        if self.segment_lungs:
            processed, _ = self.segment_lung_region(processed)
        
        if self.normalize:
            processed = self.normalize_image(processed)
        
        return processed
    
    def preprocess_batch(self, images: list) -> np.ndarray:
        """Preprocesa un lote de imágenes."""
        return np.array([self.preprocess(img) for img in images])
