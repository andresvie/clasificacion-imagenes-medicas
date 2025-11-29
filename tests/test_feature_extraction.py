"""
Tests para el módulo de extracción de características.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from feature_extraction import FeatureExtractor


class TestFeatureExtractor:
    """Tests para FeatureExtractor."""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba normalizada."""
        np.random.seed(42)
        return np.random.rand(224, 224).astype(np.float32)
    
    @pytest.fixture
    def extractor(self):
        """Crea un extractor con todas las características."""
        return FeatureExtractor(
            include_intensity=True,
            include_glcm=True,
            include_lbp=True,
            include_gabor=True,
            include_gradient=True
        )
    
    def test_intensity_features(self, extractor, sample_image):
        """Test de características de intensidad."""
        features = extractor.extract_intensity_features(sample_image)
        
        assert 'intensity_mean' in features
        assert 'intensity_std' in features
        assert 'intensity_entropy' in features
        assert 'intensity_p50' in features
        
        # Validar rangos
        assert 0 <= features['intensity_mean'] <= 1
        assert features['intensity_entropy'] >= 0
    
    def test_glcm_features(self, extractor, sample_image):
        """Test de características GLCM."""
        features = extractor.extract_glcm_features(sample_image)
        
        assert 'glcm_contrast_mean' in features
        assert 'glcm_homogeneity_mean' in features
        assert 'glcm_energy_mean' in features
        assert 'glcm_correlation_mean' in features
        
        # GLCM debe producir valores finitos
        for key, value in features.items():
            assert np.isfinite(value), f"{key} no es finito: {value}"
    
    def test_lbp_features(self, extractor, sample_image):
        """Test de características LBP."""
        features = extractor.extract_lbp_features(sample_image)
        
        assert 'lbp_mean' in features
        assert 'lbp_std' in features
        assert 'lbp_entropy' in features
        assert 'lbp_bin_0' in features
    
    def test_gabor_features(self, extractor, sample_image):
        """Test de características Gabor."""
        features = extractor.extract_gabor_features(sample_image)
        
        # Debe haber características para cada combinación frecuencia-orientación
        assert len(features) > 0
        
        # Todas deben ser finitas
        for key, value in features.items():
            assert np.isfinite(value), f"{key} no es finito"
    
    def test_gradient_features(self, extractor, sample_image):
        """Test de características de gradiente."""
        features = extractor.extract_gradient_features(sample_image)
        
        assert 'gradient_magnitude_mean' in features
        assert 'gradient_orientation_mean' in features
        assert 'laplacian_mean' in features
    
    def test_extract_all(self, extractor, sample_image):
        """Test de extracción completa."""
        features = extractor.extract_all_features(sample_image)
        
        # Debe haber muchas características
        assert len(features) > 50
        
        # Todas deben tener valores válidos
        for key, value in features.items():
            assert np.isfinite(value), f"{key} = {value}"
    
    def test_batch_extraction(self, extractor, sample_image):
        """Test de extracción por lotes."""
        images = [sample_image, sample_image, sample_image]
        features = extractor.extract_batch(images)
        
        assert features.shape[0] == 3
        assert features.shape[1] > 50
        assert not np.any(np.isnan(features))
    
    def test_feature_names(self, extractor, sample_image):
        """Test de nombres de características."""
        names = extractor.get_feature_names(sample_image)
        
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)


class TestFeatureExtractorConfigurations:
    """Tests para diferentes configuraciones."""
    
    @pytest.fixture
    def sample_image(self):
        np.random.seed(42)
        return np.random.rand(224, 224).astype(np.float32)
    
    def test_only_intensity(self, sample_image):
        """Test solo con intensidad."""
        ext = FeatureExtractor(
            include_intensity=True,
            include_glcm=False,
            include_lbp=False,
            include_gabor=False,
            include_gradient=False
        )
        features = ext.extract_all_features(sample_image)
        
        assert 'intensity_mean' in features
        assert 'glcm_contrast_mean' not in features
    
    def test_only_glcm(self, sample_image):
        """Test solo con GLCM."""
        ext = FeatureExtractor(
            include_intensity=False,
            include_glcm=True,
            include_lbp=False,
            include_gabor=False,
            include_gradient=False
        )
        features = ext.extract_all_features(sample_image)
        
        assert 'glcm_contrast_mean' in features
        assert 'intensity_mean' not in features
    
    def test_uint8_input(self, sample_image):
        """Test con imagen uint8."""
        img_uint8 = (sample_image * 255).astype(np.uint8)
        ext = FeatureExtractor()
        features = ext.extract_all_features(img_uint8)
        
        assert len(features) > 0
        for value in features.values():
            assert np.isfinite(value)


class TestFeatureConsistency:
    """Tests de consistencia de características."""
    
    def test_same_image_same_features(self):
        """La misma imagen debe producir las mismas características."""
        np.random.seed(42)
        img = np.random.rand(224, 224).astype(np.float32)
        
        ext = FeatureExtractor()
        features1 = ext.extract_all_features(img)
        features2 = ext.extract_all_features(img)
        
        for key in features1:
            assert features1[key] == features2[key], f"{key} difiere"
    
    def test_different_images_different_features(self):
        """Imágenes diferentes deben producir características diferentes."""
        np.random.seed(42)
        img1 = np.random.rand(224, 224).astype(np.float32)
        np.random.seed(123)
        img2 = np.random.rand(224, 224).astype(np.float32)
        
        ext = FeatureExtractor()
        features1 = ext.extract_all_features(img1)
        features2 = ext.extract_all_features(img2)
        
        # Al menos algunas características deben diferir
        different = sum(1 for k in features1 if features1[k] != features2[k])
        assert different > 0
