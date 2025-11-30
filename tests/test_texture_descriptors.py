"""
Tests para el módulo de descriptores de textura.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from texture_descriptors import (
    LBPDescriptor,
    GLCMDescriptor,
    GaborDescriptor,
    FirstOrderStatistics,
    TextureDescriptorExtractor
)


class TestLBPDescriptor:
    """Tests para LBPDescriptor."""
    
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
    def lbp_descriptor(self):
        """Crea un descriptor LBP con parámetros por defecto."""
        return LBPDescriptor()
    
    def test_initialization_default(self):
        """Test de inicialización con parámetros por defecto."""
        lbp = LBPDescriptor()
        assert lbp.radius == 3
        assert lbp.n_points == 24
        assert lbp.method == 'uniform'
    
    def test_initialization_custom(self):
        """Test de inicialización con parámetros personalizados."""
        lbp = LBPDescriptor(radius=5, n_points=16, method='default')
        assert lbp.radius == 5
        assert lbp.n_points == 16
        assert lbp.method == 'default'
    
    def test_compute_lbp_uint8(self, lbp_descriptor, sample_image):
        """Test de cálculo de LBP con imagen uint8."""
        lbp_image = lbp_descriptor.compute_lbp(sample_image)
        
        assert lbp_image.shape == sample_image.shape
        assert lbp_image.dtype == np.uint8
        assert lbp_image.min() >= 0
        assert lbp_image.max() <= 255
    
    def test_compute_lbp_float(self, lbp_descriptor, sample_image_float):
        """Test de cálculo de LBP con imagen normalizada."""
        lbp_image = lbp_descriptor.compute_lbp(sample_image_float)
        
        assert lbp_image.shape == sample_image_float.shape
        assert lbp_image.dtype == np.uint8
    
    def test_extract_uint8(self, lbp_descriptor, sample_image):
        """Test de extracción con imagen uint8."""
        result = lbp_descriptor.extract(sample_image)
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Debe tener bins del histograma
        assert any('lbp_bin_' in k for k in result.keys())
        
        # Debe tener estadísticas
        assert 'lbp_mean' in result
        assert 'lbp_std' in result
        assert 'lbp_entropy' in result
        
        # Validar tipos
        for value in result.values():
            assert isinstance(value, float)
            assert np.isfinite(value)
            assert value >= 0
    
    def test_extract_float(self, lbp_descriptor, sample_image_float):
        """Test de extracción con imagen normalizada."""
        result = lbp_descriptor.extract(sample_image_float)
        
        assert isinstance(result, dict)
        assert 'lbp_mean' in result
        assert 'lbp_std' in result
        assert 'lbp_entropy' in result
        
        # Validar que todos los valores son finitos
        for value in result.values():
            assert np.isfinite(value)
    
    def test_extract_different_radii(self, sample_image):
        """Test con diferentes radios."""
        for radius in [1, 3, 5]:
            lbp = LBPDescriptor(radius=radius)
            result = lbp.extract(sample_image)
            assert 'lbp_mean' in result
            assert len(result) > 0
    
    def test_extract_different_n_points(self, sample_image):
        """Test con diferentes números de puntos."""
        for n_points in [8, 16, 24]:
            lbp = LBPDescriptor(n_points=n_points)
            result = lbp.extract(sample_image)
            assert 'lbp_mean' in result
            assert len(result) > 0
    
    def test_extract_different_methods(self, sample_image):
        """Test con diferentes métodos."""
        for method in ['uniform', 'default']:
            lbp = LBPDescriptor(method=method)
            result = lbp.extract(sample_image)
            assert 'lbp_mean' in result
            assert len(result) > 0
    
    def test_extract_empty_image(self, lbp_descriptor):
        """Test con imagen vacía."""
        empty_img = np.zeros((128, 128), dtype=np.uint8)
        result = lbp_descriptor.extract(empty_img)
        
        assert isinstance(result, dict)
        assert 'lbp_mean' in result
        # En imagen vacía, la media debería ser 0 o muy pequeña
        assert result['lbp_mean'] >= 0
    
    def test_extract_uniform_image(self, lbp_descriptor):
        """Test con imagen uniforme."""
        uniform_img = np.ones((128, 128), dtype=np.uint8) * 128
        result = lbp_descriptor.extract(uniform_img)
        
        assert isinstance(result, dict)
        assert 'lbp_mean' in result
        assert np.isfinite(result['lbp_mean'])


class TestGLCMDescriptor:
    """Tests para GLCMDescriptor."""
    
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
    def glcm_descriptor(self):
        """Crea un descriptor GLCM con parámetros por defecto."""
        return GLCMDescriptor()
    
    def test_initialization_default(self):
        """Test de inicialización con parámetros por defecto."""
        glcm = GLCMDescriptor()
        assert glcm.distances == [1, 3, 5]
        assert len(glcm.angles) == 4
        assert glcm.levels == 64
    
    def test_initialization_custom(self):
        """Test de inicialización con parámetros personalizados."""
        distances = [1, 2]
        angles = [0, np.pi/2]
        glcm = GLCMDescriptor(distances=distances, angles=angles, levels=32)
        assert glcm.distances == distances
        assert glcm.angles == angles
        assert glcm.levels == 32
    
    def test_compute_glcm_from_scratch(self, glcm_descriptor, sample_image):
        """Test de cálculo de GLCM desde cero."""
        glcm_matrix = glcm_descriptor.compute_glcm_from_scratch(
            sample_image, distance=1, angle=0
        )
        
        assert glcm_matrix.shape == (glcm_descriptor.levels, glcm_descriptor.levels)
        assert np.allclose(np.sum(glcm_matrix), 1.0, atol=1e-6)  # Debe estar normalizada
        assert np.all(glcm_matrix >= 0)  # Todos los valores deben ser no negativos
    
    def test_extract_uint8(self, glcm_descriptor, sample_image):
        """Test de extracción con imagen uint8."""
        result = glcm_descriptor.extract(sample_image)
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Debe tener propiedades GLCM
        properties = ['contrast', 'dissimilarity', 'homogeneity', 
                     'energy', 'correlation', 'ASM']
        
        for prop in properties:
            assert f'glcm_{prop}_mean' in result
            assert f'glcm_{prop}_std' in result
            assert f'glcm_{prop}_min' in result
            assert f'glcm_{prop}_max' in result
        
        # Validar tipos y rangos
        for value in result.values():
            assert isinstance(value, float)
            assert np.isfinite(value)
        
        # Validar relaciones lógicas
        for prop in properties:
            mean_val = result[f'glcm_{prop}_mean']
            min_val = result[f'glcm_{prop}_min']
            max_val = result[f'glcm_{prop}_max']
            assert min_val <= mean_val <= max_val
    
    def test_extract_float(self, glcm_descriptor, sample_image_float):
        """Test de extracción con imagen normalizada."""
        result = glcm_descriptor.extract(sample_image_float)
        
        assert isinstance(result, dict)
        assert 'glcm_contrast_mean' in result
        
        # Validar que todos los valores son finitos
        for value in result.values():
            assert np.isfinite(value)
    
    def test_extract_different_distances(self, sample_image):
        """Test con diferentes distancias."""
        for distances in [[1], [1, 3], [1, 3, 5]]:
            glcm = GLCMDescriptor(distances=distances)
            result = glcm.extract(sample_image)
            assert 'glcm_contrast_mean' in result
            assert len(result) > 0
    
    def test_extract_different_angles(self, sample_image):
        """Test con diferentes ángulos."""
        angles = [0]
        glcm = GLCMDescriptor(angles=angles)
        result = glcm.extract(sample_image)
        assert 'glcm_contrast_mean' in result
    
    def test_extract_different_levels(self, sample_image):
        """Test con diferentes niveles de cuantización."""
        for levels in [32, 64, 128]:
            glcm = GLCMDescriptor(levels=levels)
            result = glcm.extract(sample_image)
            assert 'glcm_contrast_mean' in result
    
    def test_extract_empty_image(self, glcm_descriptor):
        """Test con imagen vacía."""
        empty_img = np.zeros((128, 128), dtype=np.uint8)
        result = glcm_descriptor.extract(empty_img)
        
        assert isinstance(result, dict)
        assert 'glcm_contrast_mean' in result
        # Validar que todos los valores son finitos
        for value in result.values():
            assert np.isfinite(value)
    
    def test_extract_uniform_image(self, glcm_descriptor):
        """Test con imagen uniforme."""
        uniform_img = np.ones((128, 128), dtype=np.uint8) * 128
        result = glcm_descriptor.extract(uniform_img)
        
        assert isinstance(result, dict)
        # En imagen uniforme, algunas propiedades pueden ser específicas
        assert 'glcm_contrast_mean' in result
        assert np.isfinite(result['glcm_contrast_mean'])


class TestGaborDescriptor:
    """Tests para GaborDescriptor."""
    
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
    def gabor_descriptor(self):
        """Crea un descriptor de Gabor con parámetros por defecto."""
        return GaborDescriptor()
    
    def test_initialization_default(self):
        """Test de inicialización con parámetros por defecto."""
        gabor = GaborDescriptor()
        assert len(gabor.frequencies) == 4
        assert len(gabor.orientations) == 4
        assert gabor.sigma == 1.0
    
    def test_initialization_custom(self):
        """Test de inicialización con parámetros personalizados."""
        frequencies = [0.1, 0.2]
        orientations = [0, np.pi/4]
        gabor = GaborDescriptor(frequencies=frequencies, 
                               orientations=orientations, 
                               sigma=2.0)
        assert gabor.frequencies == frequencies
        assert gabor.orientations == orientations
        assert gabor.sigma == 2.0
    
    def test_apply_filters_uint8(self, gabor_descriptor, sample_image):
        """Test de aplicación de filtros con imagen uint8."""
        responses = gabor_descriptor.apply_filters(sample_image)
        
        assert isinstance(responses, list)
        assert len(responses) == len(gabor_descriptor.frequencies) * len(gabor_descriptor.orientations)
        
        for response in responses:
            assert isinstance(response, np.ndarray)
            assert response.shape == sample_image.shape
            assert np.all(response >= 0)  # Magnitudes son no negativas
            assert np.isfinite(response).all()
    
    def test_apply_filters_float(self, gabor_descriptor, sample_image_float):
        """Test de aplicación de filtros con imagen normalizada."""
        responses = gabor_descriptor.apply_filters(sample_image_float)
        
        assert isinstance(responses, list)
        assert len(responses) > 0
        
        for response in responses:
            assert response.shape == sample_image_float.shape
            assert np.all(response >= 0)
    
    def test_extract_compact_uint8(self, gabor_descriptor, sample_image):
        """Test de extracción compacta con imagen uint8."""
        result = gabor_descriptor.extract_compact(sample_image)
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Debe tener estadísticas por filtro
        n_filters = len(gabor_descriptor.frequencies) * len(gabor_descriptor.orientations)
        for idx in range(n_filters):
            assert f'gabor_{idx}_mean' in result
            assert f'gabor_{idx}_std' in result
            assert f'gabor_{idx}_max' in result
            assert f'gabor_{idx}_var' in result
        
        # Debe tener estadísticas agregadas
        assert 'gabor_all_mean' in result
        assert 'gabor_all_std' in result
        assert 'gabor_all_max' in result
        assert 'gabor_all_min' in result
        assert 'gabor_all_var' in result
        
        # Debe tener estadísticas por frecuencia
        for f_idx in range(len(gabor_descriptor.frequencies)):
            assert f'gabor_freq_{f_idx}_mean' in result
            assert f'gabor_freq_{f_idx}_std' in result
        
        # Debe tener estadísticas por orientación
        for o_idx in range(len(gabor_descriptor.orientations)):
            assert f'gabor_orient_{o_idx}_mean' in result
            assert f'gabor_orient_{o_idx}_std' in result
        
        # Validar tipos y valores
        for value in result.values():
            assert isinstance(value, float)
            assert np.isfinite(value)
            assert value >= 0
    
    def test_extract_compact_float(self, gabor_descriptor, sample_image_float):
        """Test de extracción compacta con imagen normalizada."""
        result = gabor_descriptor.extract_compact(sample_image_float)
        
        assert isinstance(result, dict)
        assert 'gabor_all_mean' in result
        
        # Validar que todos los valores son finitos
        for value in result.values():
            assert np.isfinite(value)
    
    def test_extract_different_frequencies(self, sample_image):
        """Test con diferentes frecuencias."""
        for frequencies in [[0.1], [0.1, 0.2], [0.1, 0.2, 0.3]]:
            gabor = GaborDescriptor(frequencies=frequencies)
            result = gabor.extract_compact(sample_image)
            assert 'gabor_all_mean' in result
            assert len(result) > 0
    
    def test_extract_different_orientations(self, sample_image):
        """Test con diferentes orientaciones."""
        orientations = [0]
        gabor = GaborDescriptor(orientations=orientations)
        result = gabor.extract_compact(sample_image)
        assert 'gabor_all_mean' in result
    
    def test_extract_different_sigma(self, sample_image):
        """Test con diferentes valores de sigma."""
        for sigma in [0.5, 1.0, 2.0]:
            gabor = GaborDescriptor(sigma=sigma)
            result = gabor.extract_compact(sample_image)
            assert 'gabor_all_mean' in result
    
    def test_extract_empty_image(self, gabor_descriptor):
        """Test con imagen vacía."""
        empty_img = np.zeros((128, 128), dtype=np.uint8)
        result = gabor_descriptor.extract_compact(empty_img)
        
        assert isinstance(result, dict)
        assert 'gabor_all_mean' in result
        # En imagen vacía, las respuestas deberían ser pequeñas o cero
        assert result['gabor_all_mean'] >= 0
        assert np.isfinite(result['gabor_all_mean'])


class TestFirstOrderStatistics:
    """Tests para FirstOrderStatistics."""
    
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
    def stats_descriptor(self):
        """Crea un descriptor de estadísticas de primer orden."""
        return FirstOrderStatistics()
    
    def test_initialization(self):
        """Test de inicialización."""
        stats = FirstOrderStatistics()
        assert stats is not None
    
    def test_extract_uint8(self, stats_descriptor, sample_image):
        """Test de extracción con imagen uint8."""
        result = stats_descriptor.extract(sample_image)
        
        assert isinstance(result, dict)
        
        # Debe tener todas las estadísticas
        required_stats = [
            'stat_mean', 'stat_std', 'stat_variance', 
            'stat_skewness', 'stat_kurtosis', 'stat_entropy',
            'stat_energy', 'stat_min', 'stat_max', 'stat_range'
        ]
        
        for stat in required_stats:
            assert stat in result
        
        # Validar tipos
        for value in result.values():
            assert isinstance(value, float)
            assert np.isfinite(value)
        
        # Validar relaciones lógicas
        assert result['stat_min'] <= result['stat_mean'] <= result['stat_max']
        assert result['stat_range'] == result['stat_max'] - result['stat_min']
        assert result['stat_std'] >= 0
        assert result['stat_variance'] >= 0
        # Entropía puede ser negativa debido a normalización con density=True en histograma
        # pero debe ser un valor finito y razonable
        assert np.isfinite(result['stat_entropy'])
        assert result['stat_energy'] >= 0
    
    def test_extract_float(self, stats_descriptor, sample_image_float):
        """Test de extracción con imagen normalizada."""
        result = stats_descriptor.extract(sample_image_float)
        
        assert isinstance(result, dict)
        assert 'stat_mean' in result
        
        # Validar que todos los valores son finitos
        for value in result.values():
            assert np.isfinite(value)
        
        # Para imagen normalizada, valores deben estar en [0, 1]
        assert 0 <= result['stat_mean'] <= 1
        assert 0 <= result['stat_min'] <= 1
        assert 0 <= result['stat_max'] <= 1
    
    def test_extract_empty_image(self, stats_descriptor):
        """Test con imagen vacía."""
        empty_img = np.zeros((128, 128), dtype=np.uint8)
        result = stats_descriptor.extract(empty_img)
        
        assert isinstance(result, dict)
        assert result['stat_mean'] == 0.0
        assert result['stat_std'] == 0.0
        assert result['stat_min'] == 0.0
        assert result['stat_max'] == 0.0
        assert result['stat_range'] == 0.0
    
    def test_extract_uniform_image(self, stats_descriptor):
        """Test con imagen uniforme."""
        uniform_img = np.ones((128, 128), dtype=np.uint8) * 128
        result = stats_descriptor.extract(uniform_img)
        
        assert isinstance(result, dict)
        # En imagen uniforme, std debería ser 0
        assert result['stat_std'] == 0.0
        assert result['stat_variance'] == 0.0
        # Skewness y kurtosis pueden ser 0 o específicos para distribución uniforme
        assert np.isfinite(result['stat_skewness'])
        assert np.isfinite(result['stat_kurtosis'])
    
    def test_extract_consistency(self, stats_descriptor, sample_image):
        """Test de consistencia: misma imagen debe producir mismos resultados."""
        result1 = stats_descriptor.extract(sample_image)
        result2 = stats_descriptor.extract(sample_image)
        
        for key in result1:
            assert result1[key] == result2[key]


class TestTextureDescriptorExtractor:
    """Tests para TextureDescriptorExtractor."""
    
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
        return TextureDescriptorExtractor(
            use_lbp=True,
            use_glcm=True,
            use_gabor=True,
            use_stats=True
        )
    
    def test_initialization_all(self):
        """Test de inicialización con todos los descriptores."""
        extractor = TextureDescriptorExtractor(
            use_lbp=True,
            use_glcm=True,
            use_gabor=True,
            use_stats=True
        )
        
        assert extractor.use_lbp is True
        assert extractor.use_glcm is True
        assert extractor.use_gabor is True
        assert extractor.use_stats is True
        assert extractor.lbp_descriptor is not None
        assert extractor.glcm_descriptor is not None
        assert extractor.gabor_descriptor is not None
        assert extractor.stats_descriptor is not None
    
    def test_initialization_none(self):
        """Test de inicialización sin descriptores."""
        extractor = TextureDescriptorExtractor(
            use_lbp=False,
            use_glcm=False,
            use_gabor=False,
            use_stats=False
        )
        
        assert extractor.lbp_descriptor is None
        assert extractor.glcm_descriptor is None
        assert extractor.gabor_descriptor is None
        assert extractor.stats_descriptor is None
    
    def test_initialization_custom_kwargs(self):
        """Test de inicialización con kwargs personalizados."""
        extractor = TextureDescriptorExtractor(
            use_lbp=True,
            use_glcm=True,
            use_gabor=True,
            lbp_kwargs={'radius': 5, 'n_points': 16},
            glcm_kwargs={'levels': 32},
            gabor_kwargs={'sigma': 2.0}
        )
        
        assert extractor.lbp_descriptor.radius == 5
        assert extractor.lbp_descriptor.n_points == 16
        assert extractor.glcm_descriptor.levels == 32
        assert extractor.gabor_descriptor.sigma == 2.0
    
    def test_extract_all(self, extractor, sample_image):
        """Test de extracción con todos los descriptores."""
        features = extractor.extract(sample_image)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Debe tener características LBP
        assert any('lbp_' in k for k in features.keys())
        
        # Debe tener características GLCM
        assert any('glcm_' in k for k in features.keys())
        
        # Debe tener características Gabor
        assert any('gabor_' in k for k in features.keys())
        
        # Debe tener estadísticas de primer orden
        assert any('stat_' in k for k in features.keys())
        
        # Todas las características deben ser finitas
        for key, value in features.items():
            assert np.isfinite(value), f"{key} = {value} no es finito"
    
    def test_extract_only_lbp(self, sample_image):
        """Test de extracción solo con LBP."""
        extractor = TextureDescriptorExtractor(
            use_lbp=True,
            use_glcm=False,
            use_gabor=False,
            use_stats=False
        )
        
        features = extractor.extract(sample_image)
        
        assert any('lbp_' in k for k in features.keys())
        assert not any('glcm_' in k for k in features.keys())
        assert not any('gabor_' in k for k in features.keys())
        assert not any('stat_' in k for k in features.keys())
    
    def test_extract_only_glcm(self, sample_image):
        """Test de extracción solo con GLCM."""
        extractor = TextureDescriptorExtractor(
            use_lbp=False,
            use_glcm=True,
            use_gabor=False,
            use_stats=False
        )
        
        features = extractor.extract(sample_image)
        
        assert any('glcm_' in k for k in features.keys())
        assert not any('lbp_' in k for k in features.keys())
        assert not any('gabor_' in k for k in features.keys())
        assert not any('stat_' in k for k in features.keys())
    
    def test_extract_only_gabor(self, sample_image):
        """Test de extracción solo con Gabor."""
        extractor = TextureDescriptorExtractor(
            use_lbp=False,
            use_glcm=False,
            use_gabor=True,
            use_stats=False
        )
        
        features = extractor.extract(sample_image)
        
        assert any('gabor_' in k for k in features.keys())
        assert not any('lbp_' in k for k in features.keys())
        assert not any('glcm_' in k for k in features.keys())
        assert not any('stat_' in k for k in features.keys())
    
    def test_extract_only_stats(self, sample_image):
        """Test de extracción solo con estadísticas de primer orden."""
        extractor = TextureDescriptorExtractor(
            use_lbp=False,
            use_glcm=False,
            use_gabor=False,
            use_stats=True
        )
        
        features = extractor.extract(sample_image)
        
        assert any('stat_' in k for k in features.keys())
        assert not any('lbp_' in k for k in features.keys())
        assert not any('glcm_' in k for k in features.keys())
        assert not any('gabor_' in k for k in features.keys())
    
    def test_extract_float_image(self, extractor, sample_image_float):
        """Test con imagen normalizada."""
        features = extractor.extract(sample_image_float)
        
        assert len(features) > 0
        for key, value in features.items():
            assert np.isfinite(value), f"{key} = {value} no es finito"
    
    def test_extract_empty_image(self, extractor):
        """Test con imagen vacía."""
        empty_img = np.zeros((128, 128), dtype=np.uint8)
        features = extractor.extract(empty_img)
        
        # Debe retornar características válidas (aunque algunas sean cero)
        assert len(features) > 0
        for value in features.values():
            assert np.isfinite(value)


class TestTextureDescriptorConsistency:
    """Tests de consistencia de descriptores de textura."""
    
    def test_same_image_same_features(self):
        """La misma imagen debe producir las mismas características."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        
        extractor = TextureDescriptorExtractor()
        features1 = extractor.extract(img)
        features2 = extractor.extract(img)
        
        for key in features1:
            assert features1[key] == features2[key], \
                f"{key} difiere: {features1[key]} vs {features2[key]}"
    
    def test_different_images_different_features(self):
        """Imágenes diferentes deben producir características diferentes."""
        np.random.seed(42)
        img1 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        np.random.seed(123)
        img2 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        
        extractor = TextureDescriptorExtractor()
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
        
        extractor = TextureDescriptorExtractor()
        features_uint8 = extractor.extract(img_uint8)
        features_float = extractor.extract(img_float)
        
        # Las características deben ser similares (puede haber pequeñas diferencias numéricas)
        for key in features_uint8:
            if key in features_float:
                val1, val2 = features_uint8[key], features_float[key]
                # Permitir pequeñas diferencias numéricas
                diff = abs(val1 - val2)
                # Para características normalizadas, las diferencias pueden ser mayores
                # pero deberían ser razonables
                assert diff < 1e-2 or abs(val1) < 1e-6, \
                    f"{key} difiere significativamente: {val1} vs {val2}"

