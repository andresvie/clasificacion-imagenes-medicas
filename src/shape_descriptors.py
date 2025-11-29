"""
Descriptores de forma para análisis de imágenes médicas.

Implementa:
- Histogram of Oriented Gradients (HOG)
- Momentos de Hu (7 momentos invariantes)
- Descriptores de contorno (área, perímetro, circularidad, excentricidad)
- Fourier Shape Descriptors
"""

import numpy as np
import cv2
from skimage.feature import hog
from typing import Dict, Tuple, List, Optional


class HOGDescriptor:
    """
    Descriptor HOG (Histogram of Oriented Gradients).
    
    Captura la distribución de direcciones de gradientes locales,
    útil para detectar patrones de textura y estructura.
    """
    
    def __init__(self,
                 orientations: int = 9,
                 pixels_per_cell: Tuple[int, int] = (8, 8),
                 cells_per_block: Tuple[int, int] = (2, 2)):
        """
        Inicializa el descriptor HOG.
        
        Args:
            orientations: Número de bins de orientación (default: 9)
            pixels_per_cell: Tamaño de celda en píxeles (default: (8, 8))
            cells_per_block: Número de celdas por bloque (default: (2, 2))
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def extract(self, image: np.ndarray, visualize: bool = False) -> Dict:
        """
        Extrae descriptores HOG de la imagen.
        
        Args:
            image: Imagen en escala de grises
            visualize: Si True, retorna también la visualización
            
        Returns:
            Dict con 'n_features', 'features', y opcionalmente 'visualization'
        """
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()
        
        # Calcular HOG
        if visualize:
            features, hog_image = hog(
                img_uint8,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=True,
                feature_vector=True
            )
        else:
            features = hog(
                img_uint8,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=False,
                feature_vector=True
            )
            hog_image = None
        
        result = {
            'n_features': len(features),
            'features': features
        }
        
        if visualize and hog_image is not None:
            result['visualization'] = hog_image
        
        return result
    
    def get_statistics(self, features: np.ndarray) -> Dict:
        """
        Calcula estadísticas de los features HOG para reducir dimensionalidad.
        
        Args:
            features: Array de features HOG
            
        Returns:
            Dict con estadísticas (mean, std, min, max, etc.)
        """
        return {
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'median': float(np.median(features)),
            'q25': float(np.percentile(features, 25)),
            'q75': float(np.percentile(features, 75)),
            'skewness': float(self._skewness(features)),
            'kurtosis': float(self._kurtosis(features))
        }
    
    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        """Calcula skewness."""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return np.mean(((x - mean) / std) ** 3)
    
    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Calcula kurtosis."""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return np.mean(((x - mean) / std) ** 4) - 3.0


class HuMomentsDescriptor:
    """
    Descriptor de Momentos de Hu.
    
    Los 7 momentos de Hu son invariantes a rotación, escala y traslación.
    """
    
    def __init__(self, log_transform: bool = True):
        """
        Inicializa el descriptor de momentos de Hu.
        
        Args:
            log_transform: Si True, aplica transformación logarítmica
        """
        self.log_transform = log_transform
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extrae los 7 momentos de Hu de la imagen.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Dict con 'hu_1' a 'hu_7'
        """
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()
        
        # Calcular momentos
        moments = cv2.moments(img_uint8)
        
        # Calcular momentos de Hu
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Aplicar transformación logarítmica si se solicita
        if self.log_transform:
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        result = {}
        for i in range(1, 8):
            result[f'hu_{i}'] = float(hu_moments[i - 1])
        
        return result
    
    def explain_invariances(self) -> str:
        """
        Retorna una explicación de las invarianzas de los momentos de Hu.
        
        Returns:
            String con la explicación
        """
        return """
Momentos de Hu - Invarianzas:

Los 7 momentos de Hu son invariantes a:
- Traslación: No dependen de la posición del objeto
- Escala: No dependen del tamaño del objeto
- Rotación: No dependen de la orientación del objeto

Momento 7 es invariante a rotación pero cambia de signo con reflexión,
lo que permite distinguir entre formas especulares.

La transformación logarítmica reduce el rango de valores y mejora
la estabilidad numérica.
        """.strip()


class ContourDescriptor:
    """
    Descriptor de contornos.
    
    Extrae métricas geométricas de contornos detectados en la imagen.
    """
    
    def __init__(self, min_area: int = 100):
        """
        Inicializa el descriptor de contornos.
        
        Args:
            min_area: Área mínima para considerar un contorno (default: 100)
        """
        self.min_area = min_area
    
    def extract_contours(self, image: np.ndarray) -> List:
        """
        Extrae contornos de la imagen.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Lista de contornos detectados
        """
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()
        
        # Segmentación con Otsu
        _, binary = cv2.threshold(img_uint8, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar por área mínima
        filtered_contours = [c for c in contours 
                            if cv2.contourArea(c) >= self.min_area]
        
        return filtered_contours
    
    def extract_from_contour(self, contour) -> Dict:
        """
        Extrae descriptores de un contorno específico.
        
        Args:
            contour: Contorno de OpenCV
            
        Returns:
            Dict con descriptores del contorno
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        
        # Circularidad: 4π × área / perímetro²
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0.0
        
        # Excentricidad usando el elipse ajustado
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
            if a > 0:
                eccentricity = np.sqrt(1 - (b**2 / a**2))
            else:
                eccentricity = 0.0
        else:
            eccentricity = 0.0
        
        # Solidez (solidity): área / área convexa
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0.0
        
        # Momentos del contorno
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = 0, 0
        
        return {
            'contour_area': float(area),
            'contour_perimeter': float(perimeter),
            'contour_circularity': float(circularity),
            'contour_eccentricity': float(eccentricity),
            'contour_solidity': float(solidity),
            'contour_centroid_x': float(cx),
            'contour_centroid_y': float(cy)
        }
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extrae descriptores de contorno de la imagen.
        
        Usa el contorno más grande detectado.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Dict con descriptores de contorno
        """
        contours = self.extract_contours(image)
        
        if len(contours) == 0:
            return {
                'contour_area': 0.0,
                'contour_perimeter': 0.0,
                'contour_circularity': 0.0,
                'contour_eccentricity': 0.0,
                'contour_solidity': 0.0,
                'contour_centroid_x': 0.0,
                'contour_centroid_y': 0.0,
                'n_contours': 0
            }
        
        # Usar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        result = self.extract_from_contour(largest_contour)
        result['n_contours'] = len(contours)
        
        return result


class FourierShapeDescriptor:
    """
    Descriptor de forma basado en transformada de Fourier del contorno.
    
    Representa el contorno en el dominio de frecuencia, donde los
    coeficientes de baja frecuencia capturan la forma general y los
    de alta frecuencia capturan detalles finos.
    """
    
    def __init__(self, n_coefficients: int = 20, normalize: bool = True):
        """
        Inicializa el descriptor de Fourier.
        
        Args:
            n_coefficients: Número de coeficientes a extraer (default: 20)
            normalize: Si True, normaliza por el primer coeficiente (default: True)
        """
        self.n_coefficients = n_coefficients
        self.normalize = normalize
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extrae descriptores de Fourier del contorno.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Dict con magnitudes de Fourier y métricas agregadas
        """
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()
        
        # Segmentación y extracción de contorno
        _, binary = cv2.threshold(img_uint8, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            # Retornar valores por defecto
            result = {}
            for i in range(self.n_coefficients):
                result[f'fourier_mag_{i}'] = 0.0
            result['fourier_energy'] = 0.0
            result['fourier_mean_mag'] = 0.0
            result['fourier_low_high_ratio'] = 0.0
            return result
        
        # Usar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        contour = largest_contour.reshape(-1, 2)
        
        # Convertir contorno a señal compleja: z(t) = x(t) + i*y(t)
        contour_complex = contour[:, 0] + 1j * contour[:, 1]
        
        # Aplicar FFT
        fft_result = np.fft.fft(contour_complex)
        
        # Normalizar por el primer coeficiente (DC component) si se solicita
        if self.normalize and fft_result[0] != 0:
            fft_result = fft_result / np.abs(fft_result[0])
        
        # Calcular magnitudes
        magnitudes = np.abs(fft_result)
        
        # Tomar los primeros n_coefficients
        n = min(self.n_coefficients, len(magnitudes))
        selected_mags = magnitudes[:n]
        
        # Rellenar con ceros si es necesario
        if len(selected_mags) < self.n_coefficients:
            selected_mags = np.pad(selected_mags, 
                                 (0, self.n_coefficients - len(selected_mags)),
                                 mode='constant')
        
        # Construir resultado
        result = {}
        for i in range(self.n_coefficients):
            result[f'fourier_mag_{i}'] = float(selected_mags[i])
        
        # Métricas agregadas
        # Energía total
        result['fourier_energy'] = float(np.sum(selected_mags ** 2))
        
        # Magnitud media
        result['fourier_mean_mag'] = float(np.mean(selected_mags))
        
        # Ratio baja/alta frecuencia
        # Baja frecuencia: primeros n_coefficients//2
        # Alta frecuencia: últimos n_coefficients//2
        low_freq = selected_mags[:self.n_coefficients // 2]
        high_freq = selected_mags[self.n_coefficients // 2:]
        
        low_energy = np.sum(low_freq ** 2) if len(low_freq) > 0 else 0.0
        high_energy = np.sum(high_freq ** 2) if len(high_freq) > 0 else 0.0
        
        if high_energy > 0:
            result['fourier_low_high_ratio'] = float(low_energy / high_energy)
        else:
            result['fourier_low_high_ratio'] = 0.0
        
        return result


class ShapeDescriptorExtractor:
    """
    Extractor unificado de descriptores de forma.
    
    Permite extraer múltiples descriptores de forma de manera combinada.
    """
    
    def __init__(self,
                 use_hog: bool = True,
                 use_hu: bool = True,
                 use_contour: bool = True,
                 use_fourier: bool = True,
                 hog_kwargs: Optional[Dict] = None,
                 hu_kwargs: Optional[Dict] = None,
                 contour_kwargs: Optional[Dict] = None,
                 fourier_kwargs: Optional[Dict] = None):
        """
        Inicializa el extractor de descriptores de forma.
        
        Args:
            use_hog: Si True, incluye descriptores HOG
            use_hu: Si True, incluye momentos de Hu
            use_contour: Si True, incluye descriptores de contorno
            use_fourier: Si True, incluye descriptores de Fourier
            hog_kwargs: Parámetros para HOGDescriptor
            hu_kwargs: Parámetros para HuMomentsDescriptor
            contour_kwargs: Parámetros para ContourDescriptor
            fourier_kwargs: Parámetros para FourierShapeDescriptor
        """
        self.use_hog = use_hog
        self.use_hu = use_hu
        self.use_contour = use_contour
        self.use_fourier = use_fourier
        
        # Inicializar descriptores
        if use_hog:
            self.hog_descriptor = HOGDescriptor(**(hog_kwargs or {}))
        else:
            self.hog_descriptor = None
        
        if use_hu:
            self.hu_descriptor = HuMomentsDescriptor(**(hu_kwargs or {}))
        else:
            self.hu_descriptor = None
        
        if use_contour:
            self.contour_descriptor = ContourDescriptor(**(contour_kwargs or {}))
        else:
            self.contour_descriptor = None
        
        if use_fourier:
            self.fourier_descriptor = FourierShapeDescriptor(**(fourier_kwargs or {}))
        else:
            self.fourier_descriptor = None
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extrae todos los descriptores de forma seleccionados.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Dict con todos los descriptores de forma combinados
        """
        features = {}
        
        # HOG
        if self.use_hog and self.hog_descriptor:
            hog_result = self.hog_descriptor.extract(image, visualize=False)
            # Incluir estadísticas de HOG para reducir dimensionalidad
            hog_stats = self.hog_descriptor.get_statistics(hog_result['features'])
            for key, value in hog_stats.items():
                features[f'hog_{key}'] = value
        
        # Hu Moments
        if self.use_hu and self.hu_descriptor:
            hu_result = self.hu_descriptor.extract(image)
            features.update(hu_result)
        
        # Contour descriptors
        if self.use_contour and self.contour_descriptor:
            contour_result = self.contour_descriptor.extract(image)
            features.update(contour_result)
        
        # Fourier descriptors
        if self.use_fourier and self.fourier_descriptor:
            fourier_result = self.fourier_descriptor.extract(image)
            features.update(fourier_result)
        
        return features
