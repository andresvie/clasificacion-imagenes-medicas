"""
Descriptores de textura para análisis de imágenes médicas.

Implementa:
- Local Binary Patterns (LBP)
- Gray Level Co-occurrence Matrix (GLCM)
- Filtros de Gabor
- Estadísticas de primer orden
"""

import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from scipy import ndimage
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional


class LBPDescriptor:
    """
    Descriptor LBP (Local Binary Patterns).
    
    LBP es robusto a cambios de iluminación y captura
    patrones de textura locales característicos.
    """
    
    def __init__(self, 
                 radius: int = 3,
                 n_points: int = 24,
                 method: str = 'uniform'):
        """
        Inicializa el descriptor LBP.
        
        Args:
            radius: Radio del vecindario circular (default: 3)
            n_points: Número de puntos en el vecindario (default: 24)
            method: Método de LBP - 'uniform' o 'default' (default: 'uniform')
        """
        self.radius = radius
        self.n_points = n_points
        self.method = method
    
    def compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Calcula la imagen LBP.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Imagen LBP
        """
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()
        
        # Calcular LBP
        lbp = local_binary_pattern(img_uint8, self.n_points, self.radius, method=self.method)
        
        return lbp.astype(np.uint8)
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extrae descriptores LBP de la imagen.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Dict con descriptores LBP (histograma normalizado)
        """
        lbp_image = self.compute_lbp(image)
        
        # Histograma normalizado
        n_bins = self.n_points + 2
        hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, 
                              range=(0, n_bins), density=True)
        
        result = {}
        
        # Bins del histograma (al menos 10)
        for i in range(min(10, len(hist))):
            result[f'lbp_bin_{i}'] = float(hist[i])
        
        # Estadísticas adicionales
        result['lbp_mean'] = float(np.mean(lbp_image))
        result['lbp_std'] = float(np.std(lbp_image))
        result['lbp_entropy'] = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10)))
        
        return result


class GLCMDescriptor:
    """
    Descriptor GLCM (Gray Level Co-occurrence Matrix).
    
    GLCM captura patrones de co-ocurrencia de píxeles que son
    sensibles a infiltrados y consolidaciones pulmonares.
    """
    
    def __init__(self,
                 distances: List[int] = [1, 3, 5],
                 angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                 levels: int = 64):
        """
        Inicializa el descriptor GLCM.
        
        Args:
            distances: Distancias para calcular co-ocurrencia (default: [1, 3, 5])
            angles: Ángulos en radianes (default: [0, π/4, π/2, 3π/4])
            levels: Niveles de cuantización (default: 64)
        """
        self.distances = distances
        self.angles = angles
        self.levels = levels
    
    def compute_glcm_from_scratch(self, 
                                  image: np.ndarray,
                                  distance: int = 1,
                                  angle: float = 0) -> np.ndarray:
        """
        Calcula la matriz GLCM desde cero para visualización.
        
        Args:
            image: Imagen en escala de grises
            distance: Distancia para co-ocurrencia
            angle: Ángulo en radianes
            
        Returns:
            Matriz GLCM normalizada
        """
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()
        
        # Cuantizar imagen
        img_quantized = (img_uint8 // (256 // self.levels)).astype(np.uint8)
        img_quantized = np.clip(img_quantized, 0, self.levels - 1)
        
        # Calcular GLCM para un solo par (distance, angle)
        glcm = graycomatrix(img_quantized,
                          distances=[distance],
                          angles=[angle],
                          levels=self.levels,
                          symmetric=True,
                          normed=True)
        
        # Retornar la matriz (eliminar dimensiones singleton)
        return glcm[:, :, 0, 0]
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extrae características GLCM de la imagen.
        
        Propiedades calculadas:
        - Contrast: Mide la variación local
        - Correlation: Mide la dependencia lineal de píxeles
        - Energy (ASM): Mide la uniformidad
        - Homogeneity: Mide la cercanía de la distribución a la diagonal
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Dict con características GLCM
        """
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()
        
        # Cuantizar imagen
        img_quantized = (img_uint8 // (256 // self.levels)).astype(np.uint8)
        img_quantized = np.clip(img_quantized, 0, self.levels - 1)
        
        # Calcular GLCM
        glcm = graycomatrix(img_quantized,
                          distances=self.distances,
                          angles=self.angles,
                          levels=self.levels,
                          symmetric=True,
                          normed=True)
        
        # Propiedades a calcular
        properties = ['contrast', 'dissimilarity', 'homogeneity', 
                     'energy', 'correlation', 'ASM']
        
        result = {}
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            result[f'glcm_{prop}_mean'] = float(np.mean(values))
            result[f'glcm_{prop}_std'] = float(np.std(values))
            result[f'glcm_{prop}_min'] = float(np.min(values))
            result[f'glcm_{prop}_max'] = float(np.max(values))
        
        return result


class GaborDescriptor:
    """
    Descriptor de filtros de Gabor.
    
    Los filtros de Gabor detectan patrones orientados y texturas
    a diferentes escalas, útiles para identificar estructuras
    anormales en los pulmones.
    """
    
    def __init__(self,
                 frequencies: List[float] = [0.1, 0.2, 0.3, 0.4],
                 orientations: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                 sigma: float = 1.0):
        """
        Inicializa el descriptor de Gabor.
        
        Args:
            frequencies: Lista de frecuencias espaciales (default: [0.1, 0.2, 0.3, 0.4])
            orientations: Lista de orientaciones en radianes (default: [0, π/4, π/2, 3π/4])
            sigma: Desviación estándar del filtro Gaussiano (default: 1.0)
        """
        self.frequencies = frequencies
        self.orientations = orientations
        self.sigma = sigma
    
    def apply_filters(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Aplica el banco de filtros de Gabor a la imagen.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Lista de respuestas de filtros (magnitudes)
        """
        if image.dtype != np.uint8:
            img_float = image.astype(np.float64)
        else:
            img_float = image.astype(np.float64) / 255.0
        
        responses = []
        
        for freq in self.frequencies:
            for theta in self.orientations:
                # Aplicar filtro de Gabor
                real, imag = gabor(img_float, frequency=freq, theta=theta, 
                                  sigma_x=self.sigma, sigma_y=self.sigma)
                magnitude = np.sqrt(real**2 + imag**2)
                responses.append(magnitude)
        
        return responses
    
    def extract_compact(self, image: np.ndarray) -> Dict:
        """
        Extrae características compactas de Gabor.
        
        Calcula estadísticas agregadas de las respuestas de los filtros
        para reducir la dimensionalidad.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Dict con características compactas de Gabor
        """
        responses = self.apply_filters(image)
        
        result = {}
        
        # Estadísticas por filtro
        for idx, response in enumerate(responses):
            result[f'gabor_{idx}_mean'] = float(np.mean(response))
            result[f'gabor_{idx}_std'] = float(np.std(response))
            result[f'gabor_{idx}_max'] = float(np.max(response))
            result[f'gabor_{idx}_var'] = float(np.var(response))
        
        # Estadísticas agregadas de todos los filtros
        all_responses = np.array(responses)
        result['gabor_all_mean'] = float(np.mean(all_responses))
        result['gabor_all_std'] = float(np.std(all_responses))
        result['gabor_all_max'] = float(np.max(all_responses))
        result['gabor_all_min'] = float(np.min(all_responses))
        result['gabor_all_var'] = float(np.var(all_responses))
        
        # Estadísticas por frecuencia (promedio sobre orientaciones)
        for f_idx, freq in enumerate(self.frequencies):
            start_idx = f_idx * len(self.orientations)
            end_idx = start_idx + len(self.orientations)
            freq_responses = all_responses[start_idx:end_idx]
            result[f'gabor_freq_{f_idx}_mean'] = float(np.mean(freq_responses))
            result[f'gabor_freq_{f_idx}_std'] = float(np.std(freq_responses))
        
        # Estadísticas por orientación (promedio sobre frecuencias)
        for o_idx, orient in enumerate(self.orientations):
            orient_responses = all_responses[o_idx::len(self.orientations)]
            result[f'gabor_orient_{o_idx}_mean'] = float(np.mean(orient_responses))
            result[f'gabor_orient_{o_idx}_std'] = float(np.std(orient_responses))
        
        return result


class FirstOrderStatistics:
    """
    Descriptor de estadísticas de primer orden.
    
    Estas estadísticas capturan la distribución de intensidades
    sin considerar relaciones espaciales entre píxeles.
    """
    
    def __init__(self):
        """Inicializa el descriptor de estadísticas de primer orden."""
        pass
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Calcula estadísticas de primer orden de la imagen.
        
        Estadísticas calculadas:
        - Media: Valor promedio de intensidad
        - Desviación estándar: Dispersión de valores
        - Skewness: Asimetría de la distribución
        - Kurtosis: Curtosis (medida de "peso" de las colas)
        - Entropía: Medida de aleatoriedad/información
        - Energía: Suma de cuadrados de intensidades
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Dict con estadísticas de primer orden
        """
        if image.dtype != np.uint8:
            img_float = image.astype(np.float64)
        else:
            img_float = image.astype(np.float64) / 255.0
        
        # Calcular estadísticas básicas
        mean = np.mean(img_float)
        std = np.std(img_float)
        variance = np.var(img_float)
        
        # Skewness (asimetría)
        if std > 0:
            skewness = float(skew(img_float.flatten()))
        else:
            skewness = 0.0
        
        # Kurtosis (curtosis)
        if std > 0:
            kurt = float(kurtosis(img_float.flatten()))
        else:
            kurt = 0.0
        
        # Entropía
        hist, _ = np.histogram(img_float.flatten(), bins=256, density=True)
        hist = hist[hist > 0]  # Eliminar ceros
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        # Energía (suma de cuadrados)
        energy = float(np.sum(img_float ** 2))
        
        result = {
            'stat_mean': float(mean),
            'stat_std': float(std),
            'stat_variance': float(variance),
            'stat_skewness': float(skewness),
            'stat_kurtosis': float(kurt),
            'stat_entropy': float(entropy),
            'stat_energy': float(energy),
            'stat_min': float(np.min(img_float)),
            'stat_max': float(np.max(img_float)),
            'stat_range': float(np.max(img_float) - np.min(img_float))
        }
        
        return result


class TextureDescriptorExtractor:
    """
    Extractor unificado de descriptores de textura.
    
    Permite extraer múltiples descriptores de textura de manera combinada.
    """
    
    def __init__(self,
                 use_lbp: bool = True,
                 use_glcm: bool = True,
                 use_gabor: bool = True,
                 use_stats: bool = True,
                 lbp_kwargs: Optional[Dict] = None,
                 glcm_kwargs: Optional[Dict] = None,
                 gabor_kwargs: Optional[Dict] = None):
        """
        Inicializa el extractor de descriptores de textura.
        
        Args:
            use_lbp: Si True, incluye descriptores LBP
            use_glcm: Si True, incluye descriptores GLCM
            use_gabor: Si True, incluye descriptores de Gabor
            use_stats: Si True, incluye estadísticas de primer orden
            lbp_kwargs: Parámetros para LBPDescriptor
            glcm_kwargs: Parámetros para GLCMDescriptor
            gabor_kwargs: Parámetros para GaborDescriptor
        """
        self.use_lbp = use_lbp
        self.use_glcm = use_glcm
        self.use_gabor = use_gabor
        self.use_stats = use_stats
        
        # Inicializar descriptores
        if use_lbp:
            self.lbp_descriptor = LBPDescriptor(**(lbp_kwargs or {}))
        else:
            self.lbp_descriptor = None
        
        if use_glcm:
            self.glcm_descriptor = GLCMDescriptor(**(glcm_kwargs or {}))
        else:
            self.glcm_descriptor = None
        
        if use_gabor:
            self.gabor_descriptor = GaborDescriptor(**(gabor_kwargs or {}))
        else:
            self.gabor_descriptor = None
        
        if use_stats:
            self.stats_descriptor = FirstOrderStatistics()
        else:
            self.stats_descriptor = None
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extrae todos los descriptores de textura seleccionados.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Dict con todos los descriptores de textura combinados
        """
        features = {}
        
        # LBP
        if self.use_lbp and self.lbp_descriptor:
            lbp_result = self.lbp_descriptor.extract(image)
            features.update(lbp_result)
        
        # GLCM
        if self.use_glcm and self.glcm_descriptor:
            glcm_result = self.glcm_descriptor.extract(image)
            features.update(glcm_result)
        
        # Gabor
        if self.use_gabor and self.gabor_descriptor:
            gabor_result = self.gabor_descriptor.extract_compact(image)
            features.update(gabor_result)
        
        # First-order statistics
        if self.use_stats and self.stats_descriptor:
            stats_result = self.stats_descriptor.extract(image)
            features.update(stats_result)
        
        return features
