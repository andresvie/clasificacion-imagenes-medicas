"""
Extracción de características para clasificación de radiografías.

Incluye marcadores propios basados en:
- Estadísticas de intensidad
- Características de textura (GLCM, LBP)
- Características de forma y estructura
- Características frecuenciales
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from scipy import ndimage
from typing import Dict, List


class FeatureExtractor:
    """
    Extractor de características para radiografías de tórax.
    
    Implementa múltiples descriptores diseñados para capturar
    patrones relevantes en la detección de neumonía:
    
    1. Estadísticas de intensidad: Capturan cambios globales de brillo
    2. Características de textura: Detectan patrones de infiltrados
    3. Características estructurales: Analizan la estructura pulmonar
    4. Características frecuenciales: Capturan patrones periódicos
    """
    
    def __init__(self, 
                 include_intensity: bool = True,
                 include_glcm: bool = True,
                 include_lbp: bool = True,
                 include_gabor: bool = True,
                 include_gradient: bool = True):
        """
        Inicializa el extractor de características.
        
        Args:
            include_intensity: Incluir estadísticas de intensidad
            include_glcm: Incluir características GLCM
            include_lbp: Incluir Local Binary Patterns
            include_gabor: Incluir filtros de Gabor
            include_gradient: Incluir características de gradiente
        """
        self.include_intensity = include_intensity
        self.include_glcm = include_glcm
        self.include_lbp = include_lbp
        self.include_gabor = include_gabor
        self.include_gradient = include_gradient
        
        # Parámetros para características
        self.glcm_distances = [1, 3, 5]
        self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.lbp_radius = 3
        self.lbp_n_points = 24
        self.gabor_frequencies = [0.1, 0.2, 0.3, 0.4]
        self.gabor_orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    def extract_intensity_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae estadísticas de intensidad.
        
        Justificación: La neumonía causa opacidades que alteran
        la distribución de intensidades en la radiografía.
        """
        features = {}
        
        # Estadísticas básicas
        features['intensity_mean'] = np.mean(image)
        features['intensity_std'] = np.std(image)
        features['intensity_skewness'] = self._skewness(image)
        features['intensity_kurtosis'] = self._kurtosis(image)
        
        # Percentiles
        features['intensity_p10'] = np.percentile(image, 10)
        features['intensity_p25'] = np.percentile(image, 25)
        features['intensity_p50'] = np.percentile(image, 50)
        features['intensity_p75'] = np.percentile(image, 75)
        features['intensity_p90'] = np.percentile(image, 90)
        
        # Rango y contraste
        features['intensity_range'] = np.max(image) - np.min(image)
        features['intensity_iqr'] = features['intensity_p75'] - features['intensity_p25']
        
        # Entropía
        hist, _ = np.histogram(image.flatten(), bins=256, density=False)
        # Normalizar a probabilidades
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        features['intensity_entropy'] = -np.sum(hist * np.log2(hist))
        
        return features
    
    def extract_glcm_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características de textura usando GLCM.
        
        Justificación: GLCM captura patrones de co-ocurrencia de píxeles
        que son sensibles a infiltrados y consolidaciones pulmonares.
        """
        features = {}
        
        # Cuantizar imagen a 64 niveles para GLCM
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        img_quantized = (img_uint8 // 4).astype(np.uint8)
        
        # Calcular GLCM
        glcm = graycomatrix(img_quantized, 
                          distances=self.glcm_distances,
                          angles=self.glcm_angles,
                          levels=64, 
                          symmetric=True, 
                          normed=True)
        
        # Extraer propiedades
        properties = ['contrast', 'dissimilarity', 'homogeneity', 
                     'energy', 'correlation', 'ASM']
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = np.mean(values)
            features[f'glcm_{prop}_std'] = np.std(values)
        
        return features
    
    def extract_lbp_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características usando Local Binary Patterns.
        
        Justificación: LBP es robusto a cambios de iluminación
        y captura patrones de textura locales característicos
        de diferentes condiciones pulmonares.
        """
        features = {}
        
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        # Calcular LBP
        lbp = local_binary_pattern(img_uint8, 
                                   self.lbp_n_points, 
                                   self.lbp_radius, 
                                   method='uniform')
        
        # Histograma normalizado
        n_bins = self.lbp_n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, 
                              range=(0, n_bins), density=True)
        
        # Estadísticas del histograma
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_std'] = np.std(lbp)
        features['lbp_entropy'] = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        
        # Bins más relevantes
        for i, val in enumerate(hist[:10]):
            features[f'lbp_bin_{i}'] = val
        
        return features
    
    def extract_gabor_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características usando filtros de Gabor.
        
        Justificación: Los filtros de Gabor detectan patrones
        orientados y texturas a diferentes escalas, útiles para
        identificar estructuras anormales en los pulmones.
        """
        features = {}
        
        if image.dtype != np.uint8:
            img_float = image.astype(np.float64)
        else:
            img_float = image.astype(np.float64) / 255.0
        
        for freq in self.gabor_frequencies:
            for theta in self.gabor_orientations:
                real, imag = gabor(img_float, frequency=freq, theta=theta)
                magnitude = np.sqrt(real**2 + imag**2)
                
                key = f'gabor_f{freq:.1f}_t{theta:.2f}'
                features[f'{key}_mean'] = np.mean(magnitude)
                features[f'{key}_std'] = np.std(magnitude)
        
        return features
    
    def extract_gradient_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características basadas en gradientes.
        
        Justificación: Los gradientes capturan bordes y transiciones
        que son alterados por consolidaciones pulmonares.
        """
        features = {}
        
        if image.dtype != np.uint8:
            img_float = image.astype(np.float64)
        else:
            img_float = image.astype(np.float64) / 255.0
        
        # Gradientes de Sobel
        grad_x = ndimage.sobel(img_float, axis=1)
        grad_y = ndimage.sobel(img_float, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)
        
        # Estadísticas de magnitud
        features['gradient_magnitude_mean'] = np.mean(magnitude)
        features['gradient_magnitude_std'] = np.std(magnitude)
        features['gradient_magnitude_max'] = np.max(magnitude)
        
        # Estadísticas de orientación
        features['gradient_orientation_mean'] = np.mean(orientation)
        features['gradient_orientation_std'] = np.std(orientation)
        
        # Laplaciano (detecta bordes)
        laplacian = ndimage.laplace(img_float)
        features['laplacian_mean'] = np.mean(np.abs(laplacian))
        features['laplacian_std'] = np.std(laplacian)
        
        return features
    
    def extract_all_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae todas las características habilitadas.
        
        Args:
            image: Imagen preprocesada
            
        Returns:
            Dict con todas las características
        """
        features = {}
        
        if self.include_intensity:
            features.update(self.extract_intensity_features(image))
        
        if self.include_glcm:
            features.update(self.extract_glcm_features(image))
        
        if self.include_lbp:
            features.update(self.extract_lbp_features(image))
        
        if self.include_gabor:
            features.update(self.extract_gabor_features(image))
        
        if self.include_gradient:
            features.update(self.extract_gradient_features(image))
        
        return features
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extrae características de un lote de imágenes.
        
        Args:
            images: Lista de imágenes preprocesadas
            
        Returns:
            np.ndarray de forma (n_samples, n_features)
        """
        all_features = []
        for img in images:
            features = self.extract_all_features(img)
            all_features.append(list(features.values()))
        
        return np.array(all_features)
    
    def get_feature_names(self, image: np.ndarray) -> List[str]:
        """Retorna los nombres de las características."""
        features = self.extract_all_features(image)
        return list(features.keys())
    
    @staticmethod
    def _skewness(data: np.ndarray) -> float:
        """Calcula el coeficiente de asimetría."""
        n = data.size
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum(((data - mean) / std) ** 3) / n
    
    @staticmethod
    def _kurtosis(data: np.ndarray) -> float:
        """Calcula la curtosis."""
        n = data.size
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.sum(((data - mean) / std) ** 4) / n - 3
