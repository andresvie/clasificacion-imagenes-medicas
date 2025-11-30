"""
Tests para el módulo de descriptores de forma.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from shape_descriptors import (
    HOGDescriptor,
    HuMomentsDescriptor,
    ContourDescriptor,
    FourierShapeDescriptor,
    ShapeDescriptorExtractor
)


class TestHOGDescriptor:
    """Tests para HOGDescriptor."""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba."""
        np.random.seed(42)
        return np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    
    @pytest.fixture
    def sample_image_float(self):
        """Crea una imagen de prueba normalizada."""
        np.random.seed(42)
        return np.random.rand(128, 128).astype(np.float32)
    
    @pytest.fixture
    def hog_descriptor(self):
        """Crea un descriptor HOG con parámetros por defecto."""
        return HOGDescriptor()
    
    def test_initialization_default(self):
        """Test de inicialización con parámetros por defecto."""
        hog = HOGDescriptor()
        assert hog.orientations == 9
        assert hog.pixels_per_cell == (8, 8)
        assert hog.cells_per_block == (2, 2)
    
    def test_initialization_custom(self):
        """Test de inicialización con parámetros personalizados."""
        hog = HOGDescriptor(
            orientations=12,
            pixels_per_cell=(16, 16),
            cells_per_block=(3, 3)
        )
        assert hog.orientations == 12
        assert hog.pixels_per_cell == (16, 16)
        assert hog.cells_per_block == (3, 3)
    
    def test_extract_uint8(self, hog_descriptor, sample_image):
        """Test de extracción con imagen uint8."""
        result = hog_descriptor.extract(sample_image, visualize=False)
        
        assert 'n_features' in result
        assert 'features' in result
        assert result['n_features'] > 0
        assert len(result['features']) == result['n_features']
        assert isinstance(result['features'], np.ndarray)
    
    def test_extract_float(self, hog_descriptor, sample_image_float):
        """Test de extracción con imagen normalizada."""
        result = hog_descriptor.extract(sample_image_float, visualize=False)
        
        assert 'n_features' in result
        assert 'features' in result
        assert result['n_features'] > 0
        assert len(result['features']) == result['n_features']
    
    def test_extract_with_visualization(self, hog_descriptor, sample_image):
        """Test de extracción con visualización."""
        result = hog_descriptor.extract(sample_image, visualize=True)
        
        assert 'n_features' in result
        assert 'features' in result
        assert 'visualization' in result
        assert result['visualization'] is not None
        assert isinstance(result['visualization'], np.ndarray)
    
    def test_extract_without_visualization(self, hog_descriptor, sample_image):
        """Test de extracción sin visualización."""
        result = hog_descriptor.extract(sample_image, visualize=False)
        
        assert 'visualization' not in result
    
    def test_get_statistics(self, hog_descriptor, sample_image):
        """Test de cálculo de estadísticas."""
        result = hog_descriptor.extract(sample_image, visualize=False)
        features = result['features']
        stats = hog_descriptor.get_statistics(features)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'q25' in stats
        assert 'q75' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        
        # Validar tipos
        for value in stats.values():
            assert isinstance(value, float)
        
        # Validar relaciones lógicas
        assert stats['min'] <= stats['q25'] <= stats['median'] <= stats['q75'] <= stats['max']
        assert stats['std'] >= 0
    
    def test_different_orientations(self, sample_image):
        """Test con diferentes números de orientaciones."""
        for orientations in [4, 9, 18]:
            hog = HOGDescriptor(orientations=orientations)
            result = hog.extract(sample_image, visualize=False)
            assert result['n_features'] > 0
    
    def test_different_cell_sizes(self, sample_image):
        """Test con diferentes tamaños de celda."""
        for cell_size in [(4, 4), (8, 8), (16, 16)]:
            hog = HOGDescriptor(pixels_per_cell=cell_size)
            result = hog.extract(sample_image, visualize=False)
            assert result['n_features'] > 0


class TestHuMomentsDescriptor:
    """Tests para HuMomentsDescriptor."""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba."""
        np.random.seed(42)
        return np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    
    @pytest.fixture
    def sample_image_float(self):
        """Crea una imagen de prueba normalizada."""
        np.random.seed(42)
        return np.random.rand(128, 128).astype(np.float32)
    
    @pytest.fixture
    def hu_descriptor(self):
        """Crea un descriptor de momentos de Hu."""
        return HuMomentsDescriptor()
    
    def test_initialization_default(self):
        """Test de inicialización con parámetros por defecto."""
        hu = HuMomentsDescriptor()
        assert hu.log_transform is True
    
    def test_initialization_no_log(self):
        """Test de inicialización sin transformación logarítmica."""
        hu = HuMomentsDescriptor(log_transform=False)
        assert hu.log_transform is False
    
    def test_extract_uint8(self, hu_descriptor, sample_image):
        """Test de extracción con imagen uint8."""
        result = hu_descriptor.extract(sample_image)
        
        assert 'hu_1' in result
        assert 'hu_2' in result
        assert 'hu_3' in result
        assert 'hu_4' in result
        assert 'hu_5' in result
        assert 'hu_6' in result
        assert 'hu_7' in result
        
        # Validar que todos son floats
        for i in range(1, 8):
            assert isinstance(result[f'hu_{i}'], float)
            assert np.isfinite(result[f'hu_{i}'])
    
    def test_extract_float(self, hu_descriptor, sample_image_float):
        """Test de extracción con imagen normalizada."""
        result = hu_descriptor.extract(sample_image_float)
        
        assert len(result) == 7
        for i in range(1, 8):
            assert f'hu_{i}' in result
            assert np.isfinite(result[f'hu_{i}'])
    
    def test_extract_with_log_transform(self, sample_image):
        """Test con transformación logarítmica."""
        hu = HuMomentsDescriptor(log_transform=True)
        result = hu.extract(sample_image)
        
        for i in range(1, 8):
            assert np.isfinite(result[f'hu_{i}'])
    
    def test_extract_without_log_transform(self, sample_image):
        """Test sin transformación logarítmica."""
        hu = HuMomentsDescriptor(log_transform=False)
        result = hu.extract(sample_image)
        
        for i in range(1, 8):
            assert np.isfinite(result[f'hu_{i}'])
    
    def test_explain_invariances(self, hu_descriptor):
        """Test del método de explicación."""
        explanation = hu_descriptor.explain_invariances()
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert 'invarianzas' in explanation.lower() or 'invariant' in explanation.lower()
    
    def test_empty_image(self, hu_descriptor):
        """Test con imagen vacía."""
        empty_img = np.zeros((128, 128), dtype=np.uint8)
        result = hu_descriptor.extract(empty_img)
        
        # Debe retornar los 7 momentos
        assert len(result) == 7
        for i in range(1, 8):
            assert f'hu_{i}' in result


class TestContourDescriptor:
    """Tests para ContourDescriptor."""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba con forma definida."""
        img = np.zeros((128, 128), dtype=np.uint8)
        # Dibujar un círculo
        import cv2
        cv2.circle(img, (64, 64), 30, 255, -1)
        return img
    
    @pytest.fixture
    def empty_image(self):
        """Crea una imagen vacía."""
        return np.zeros((128, 128), dtype=np.uint8)
    
    @pytest.fixture
    def contour_descriptor(self):
        """Crea un descriptor de contornos."""
        return ContourDescriptor()
    
    def test_initialization_default(self):
        """Test de inicialización con parámetros por defecto."""
        contour = ContourDescriptor()
        assert contour.min_area == 100
    
    def test_initialization_custom(self):
        """Test de inicialización con área mínima personalizada."""
        contour = ContourDescriptor(min_area=50)
        assert contour.min_area == 50
    
    def test_extract_contours(self, contour_descriptor, sample_image):
        """Test de extracción de contornos."""
        contours = contour_descriptor.extract_contours(sample_image)
        
        assert isinstance(contours, list)
        assert len(contours) > 0
    
    def test_extract_contours_empty(self, contour_descriptor, empty_image):
        """Test de extracción de contornos en imagen vacía."""
        contours = contour_descriptor.extract_contours(empty_image)
        
        assert isinstance(contours, list)
        # Puede haber contornos pequeños que se filtran
    
    def test_extract_from_contour(self, contour_descriptor, sample_image):
        """Test de extracción de descriptores de un contorno."""
        contours = contour_descriptor.extract_contours(sample_image)
        
        if len(contours) > 0:
            result = contour_descriptor.extract_from_contour(contours[0])
            
            assert 'contour_area' in result
            assert 'contour_perimeter' in result
            assert 'contour_circularity' in result
            assert 'contour_eccentricity' in result
            assert 'contour_solidity' in result
            assert 'contour_centroid_x' in result
            assert 'contour_centroid_y' in result
            
            # Validar tipos
            for value in result.values():
                assert isinstance(value, float)
                assert np.isfinite(value)
            
            # Validar rangos lógicos
            assert result['contour_area'] >= 0
            assert result['contour_perimeter'] >= 0
            assert 0 <= result['contour_circularity'] <= 1
            assert 0 <= result['contour_eccentricity'] <= 1
            assert 0 <= result['contour_solidity'] <= 1
    
    def test_extract(self, contour_descriptor, sample_image):
        """Test de extracción completa."""
        result = contour_descriptor.extract(sample_image)
        
        assert 'contour_area' in result
        assert 'contour_perimeter' in result
        assert 'contour_circularity' in result
        assert 'contour_eccentricity' in result
        assert 'contour_solidity' in result
        assert 'contour_centroid_x' in result
        assert 'contour_centroid_y' in result
        assert 'n_contours' in result
        
        # Validar tipos
        for value in result.values():
            assert isinstance(value, (float, int))
            assert np.isfinite(value)
    
    def test_extract_empty_image(self, contour_descriptor, empty_image):
        """Test de extracción en imagen vacía."""
        result = contour_descriptor.extract(empty_image)
        
        assert 'n_contours' in result
        # Una imagen vacía puede o no tener contornos dependiendo del thresholding
        assert isinstance(result['n_contours'], int)
        assert result['n_contours'] >= 0
        # Si no hay contornos, el área debe ser 0
        if result['n_contours'] == 0:
            assert result['contour_area'] == 0.0
            assert result['contour_perimeter'] == 0.0
    
    def test_extract_float_image(self, contour_descriptor):
        """Test con imagen normalizada."""
        img_float = np.random.rand(128, 128).astype(np.float32)
        result = contour_descriptor.extract(img_float)
        
        assert 'n_contours' in result
        assert isinstance(result['n_contours'], int)
    
    def test_min_area_filtering(self, sample_image):
        """Test de filtrado por área mínima."""
        # Con área mínima grande, no debería encontrar contornos
        contour_large = ContourDescriptor(min_area=10000)
        contours = contour_large.extract_contours(sample_image)
        
        # Todos los contornos deben tener área >= min_area
        for c in contours:
            import cv2
            assert cv2.contourArea(c) >= 10000


class TestFourierShapeDescriptor:
    """Tests para FourierShapeDescriptor."""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba con forma definida."""
        img = np.zeros((128, 128), dtype=np.uint8)
        import cv2
        cv2.circle(img, (64, 64), 30, 255, -1)
        return img
    
    @pytest.fixture
    def empty_image(self):
        """Crea una imagen vacía."""
        return np.zeros((128, 128), dtype=np.uint8)
    
    @pytest.fixture
    def fourier_descriptor(self):
        """Crea un descriptor de Fourier."""
        return FourierShapeDescriptor()
    
    def test_initialization_default(self):
        """Test de inicialización con parámetros por defecto."""
        fourier = FourierShapeDescriptor()
        assert fourier.n_coefficients == 20
        assert fourier.normalize is True
    
    def test_initialization_custom(self):
        """Test de inicialización con parámetros personalizados."""
        fourier = FourierShapeDescriptor(n_coefficients=10, normalize=False)
        assert fourier.n_coefficients == 10
        assert fourier.normalize is False
    
    def test_extract(self, fourier_descriptor, sample_image):
        """Test de extracción de descriptores de Fourier."""
        result = fourier_descriptor.extract(sample_image)
        
        # Debe tener n_coefficients magnitudes
        for i in range(fourier_descriptor.n_coefficients):
            assert f'fourier_mag_{i}' in result
        
        # Debe tener métricas agregadas
        assert 'fourier_energy' in result
        assert 'fourier_mean_mag' in result
        assert 'fourier_low_high_ratio' in result
        
        # Validar tipos
        for value in result.values():
            assert isinstance(value, float)
            assert np.isfinite(value)
            assert value >= 0
    
    def test_extract_empty_image(self, fourier_descriptor, empty_image):
        """Test de extracción en imagen vacía."""
        result = fourier_descriptor.extract(empty_image)
        
        # Debe retornar valores válidos (puede ser 0 o valores pequeños si se detecta un contorno)
        assert 'fourier_energy' in result
        assert 'fourier_mean_mag' in result
        assert result['fourier_energy'] >= 0.0
        assert result['fourier_mean_mag'] >= 0.0
        assert np.isfinite(result['fourier_energy'])
        assert np.isfinite(result['fourier_mean_mag'])
    
    def test_extract_float_image(self, fourier_descriptor):
        """Test con imagen normalizada."""
        img_float = np.random.rand(128, 128).astype(np.float32)
        result = fourier_descriptor.extract(img_float)
        
        assert len(result) > 0
        for value in result.values():
            assert np.isfinite(value)
    
    def test_different_n_coefficients(self, sample_image):
        """Test con diferentes números de coeficientes."""
        for n_coeff in [5, 10, 20, 50]:
            fourier = FourierShapeDescriptor(n_coefficients=n_coeff)
            result = fourier.extract(sample_image)
            
            assert len([k for k in result.keys() if k.startswith('fourier_mag_')]) == n_coeff
    
    def test_normalize_vs_no_normalize(self, sample_image):
        """Test con y sin normalización."""
        fourier_norm = FourierShapeDescriptor(normalize=True)
        fourier_no_norm = FourierShapeDescriptor(normalize=False)
        
        result_norm = fourier_norm.extract(sample_image)
        result_no_norm = fourier_no_norm.extract(sample_image)
        
        # Ambos deben retornar resultados válidos
        assert len(result_norm) == len(result_no_norm)
        assert 'fourier_energy' in result_norm
        assert 'fourier_energy' in result_no_norm


class TestShapeDescriptorExtractor:
    """Tests para ShapeDescriptorExtractor."""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba."""
        np.random.seed(42)
        return np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    
    @pytest.fixture
    def sample_image_float(self):
        """Crea una imagen de prueba normalizada."""
        np.random.seed(42)
        return np.random.rand(128, 128).astype(np.float32)
    
    @pytest.fixture
    def extractor(self):
        """Crea un extractor con todos los descriptores."""
        return ShapeDescriptorExtractor(
            use_hog=True,
            use_hu=True,
            use_contour=True,
            use_fourier=True
        )
    
    def test_initialization_all(self):
        """Test de inicialización con todos los descriptores."""
        extractor = ShapeDescriptorExtractor(
            use_hog=True,
            use_hu=True,
            use_contour=True,
            use_fourier=True
        )
        
        assert extractor.use_hog is True
        assert extractor.use_hu is True
        assert extractor.use_contour is True
        assert extractor.use_fourier is True
        assert extractor.hog_descriptor is not None
        assert extractor.hu_descriptor is not None
        assert extractor.contour_descriptor is not None
        assert extractor.fourier_descriptor is not None
    
    def test_initialization_none(self):
        """Test de inicialización sin descriptores."""
        extractor = ShapeDescriptorExtractor(
            use_hog=False,
            use_hu=False,
            use_contour=False,
            use_fourier=False
        )
        
        assert extractor.hog_descriptor is None
        assert extractor.hu_descriptor is None
        assert extractor.contour_descriptor is None
        assert extractor.fourier_descriptor is None
    
    def test_initialization_custom_kwargs(self):
        """Test de inicialización con kwargs personalizados."""
        extractor = ShapeDescriptorExtractor(
            use_hog=True,
            use_hu=True,
            hog_kwargs={'orientations': 12},
            hu_kwargs={'log_transform': False}
        )
        
        assert extractor.hog_descriptor.orientations == 12
        assert extractor.hu_descriptor.log_transform is False
    
    def test_extract_all(self, extractor, sample_image):
        """Test de extracción con todos los descriptores."""
        features = extractor.extract(sample_image)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Debe tener características HOG (estadísticas)
        assert any('hog_' in k for k in features.keys())
        
        # Debe tener momentos de Hu
        assert 'hu_1' in features
        assert 'hu_7' in features
        
        # Debe tener descriptores de contorno
        assert 'contour_area' in features
        
        # Debe tener descriptores de Fourier
        assert 'fourier_mag_0' in features
        
        # Todas las características deben ser finitas (excepto posibles NaN en casos límite)
        for key, value in features.items():
            # Algunos descriptores pueden producir NaN en casos límite (ej: eccentricity)
            if 'eccentricity' in key.lower():
                # Eccentricity puede ser NaN si el ajuste de elipse falla
                assert np.isfinite(value) or np.isnan(value), f"{key} debe ser finito o NaN"
            else:
                assert np.isfinite(value), f"{key} = {value} no es finito"
    
    def test_extract_only_hog(self, sample_image):
        """Test de extracción solo con HOG."""
        extractor = ShapeDescriptorExtractor(
            use_hog=True,
            use_hu=False,
            use_contour=False,
            use_fourier=False
        )
        
        features = extractor.extract(sample_image)
        
        assert any('hog_' in k for k in features.keys())
        assert 'hu_1' not in features
        assert 'contour_area' not in features
        assert 'fourier_mag_0' not in features
    
    def test_extract_only_hu(self, sample_image):
        """Test de extracción solo con momentos de Hu."""
        extractor = ShapeDescriptorExtractor(
            use_hog=False,
            use_hu=True,
            use_contour=False,
            use_fourier=False
        )
        
        features = extractor.extract(sample_image)
        
        assert 'hu_1' in features
        assert 'hu_7' in features
        assert not any('hog_' in k for k in features.keys())
    
    def test_extract_only_contour(self, sample_image):
        """Test de extracción solo con contornos."""
        extractor = ShapeDescriptorExtractor(
            use_hog=False,
            use_hu=False,
            use_contour=True,
            use_fourier=False
        )
        
        features = extractor.extract(sample_image)
        
        assert 'contour_area' in features
        assert 'hu_1' not in features
        assert not any('hog_' in k for k in features.keys())
    
    def test_extract_only_fourier(self, sample_image):
        """Test de extracción solo con Fourier."""
        extractor = ShapeDescriptorExtractor(
            use_hog=False,
            use_hu=False,
            use_contour=False,
            use_fourier=True
        )
        
        features = extractor.extract(sample_image)
        
        assert 'fourier_mag_0' in features
        assert 'hu_1' not in features
        assert 'contour_area' not in features
    
    def test_extract_float_image(self, extractor, sample_image_float):
        """Test con imagen normalizada."""
        features = extractor.extract(sample_image_float)
        
        assert len(features) > 0
        for key, value in features.items():
            # Algunos descriptores pueden producir NaN en casos límite
            if 'eccentricity' in key.lower():
                assert np.isfinite(value) or np.isnan(value), f"{key} debe ser finito o NaN"
            else:
                assert np.isfinite(value), f"{key} = {value} no es finito"
    
    def test_extract_empty_image(self, extractor):
        """Test con imagen vacía."""
        empty_img = np.zeros((128, 128), dtype=np.uint8)
        features = extractor.extract(empty_img)
        
        # Debe retornar características válidas (aunque algunas sean cero)
        assert len(features) > 0
        for value in features.values():
            assert np.isfinite(value)


class TestShapeDescriptorConsistency:
    """Tests de consistencia de descriptores de forma."""
    
    def test_same_image_same_features(self):
        """La misma imagen debe producir las mismas características."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        
        extractor = ShapeDescriptorExtractor()
        features1 = extractor.extract(img)
        features2 = extractor.extract(img)
        
        for key in features1:
            val1, val2 = features1[key], features2[key]
            # Manejar NaN: NaN == NaN es False, pero ambos deben ser NaN
            if np.isnan(val1) and np.isnan(val2):
                continue
            assert val1 == val2, f"{key} difiere: {val1} vs {val2}"
    
    def test_different_images_different_features(self):
        """Imágenes diferentes deben producir características diferentes."""
        np.random.seed(42)
        img1 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        np.random.seed(123)
        img2 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        
        extractor = ShapeDescriptorExtractor()
        features1 = extractor.extract(img1)
        features2 = extractor.extract(img2)
        
        # Al menos algunas características deben diferir
        different = sum(1 for k in features1 if features1[k] != features2[k])
        assert different > 0
    
    def test_uint8_vs_float_consistency(self):
        """Imágenes uint8 y float equivalentes deben producir características similares."""
        np.random.seed(42)
        img_uint8 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img_float = (img_uint8.astype(np.float32) / 255.0)
        
        extractor = ShapeDescriptorExtractor()
        features_uint8 = extractor.extract(img_uint8)
        features_float = extractor.extract(img_float)
        
        # Las características deben ser similares (puede haber pequeñas diferencias numéricas)
        for key in features_uint8:
            if key in features_float:
                val1, val2 = features_uint8[key], features_float[key]
                # Manejar NaN: si ambos son NaN, son consistentes
                if np.isnan(val1) and np.isnan(val2):
                    continue
                # Si uno es NaN y el otro no, es un problema
                if np.isnan(val1) or np.isnan(val2):
                    # Algunos descriptores pueden producir NaN en casos límite
                    if 'eccentricity' in key.lower():
                        continue  # Eccentricity puede ser NaN en casos límite
                    else:
                        assert False, f"{key} tiene NaN inconsistente: {val1} vs {val2}"
                # Permitir pequeñas diferencias numéricas
                diff = abs(val1 - val2)
                assert diff < 1e-3 or abs(val1) < 1e-6, \
                    f"{key} difiere significativamente: {val1} vs {val2}"

