"""
Tests para el módulo de preprocesamiento.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import XRayPreprocessor


class TestXRayPreprocessor:
    """Tests para XRayPreprocessor."""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba."""
        np.random.seed(42)
        return np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    
    @pytest.fixture
    def preprocessor(self):
        """Crea un preprocesador con configuración por defecto."""
        return XRayPreprocessor(
            target_size=(224, 224),
            apply_clahe=True,
            clahe_clip_limit=2.0,
            segment_lungs=False,
            normalize=True
        )
    
    def test_resize(self, preprocessor, sample_image):
        """Test de redimensionamiento."""
        resized = preprocessor.resize(sample_image)
        assert resized.shape == (224, 224)
    
    def test_clahe_enhancement(self, preprocessor, sample_image):
        """Test de mejora de contraste CLAHE."""
        resized = preprocessor.resize(sample_image)
        enhanced = preprocessor.apply_clahe_enhancement(resized)
        
        assert enhanced.shape == resized.shape
        assert enhanced.dtype == np.uint8
        # CLAHE debe cambiar la imagen
        assert not np.array_equal(enhanced, resized)
    
    def test_normalize(self, preprocessor, sample_image):
        """Test de normalización."""
        resized = preprocessor.resize(sample_image)
        normalized = preprocessor.normalize_image(resized)
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_full_pipeline(self, preprocessor, sample_image):
        """Test del pipeline completo."""
        processed = preprocessor.preprocess(sample_image)
        
        assert processed.shape == (224, 224)
        assert processed.dtype == np.float32
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0
    
    def test_batch_processing(self, preprocessor, sample_image):
        """Test de procesamiento por lotes."""
        images = [sample_image, sample_image, sample_image]
        batch = preprocessor.preprocess_batch(images)
        
        assert batch.shape == (3, 224, 224)
        assert batch.dtype == np.float32
    
    def test_different_input_sizes(self, preprocessor):
        """Test con diferentes tamaños de entrada."""
        sizes = [(256, 256), (512, 512), (1024, 768), (300, 400)]
        
        for h, w in sizes:
            img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
            processed = preprocessor.preprocess(img)
            assert processed.shape == (224, 224)
    
    def test_clahe_parameters(self, sample_image):
        """Test con diferentes parámetros CLAHE."""
        clip_limits = [1.0, 2.0, 4.0]
        tile_grids = [(4, 4), (8, 8), (16, 16)]
        
        for clip in clip_limits:
            for grid in tile_grids:
                prep = XRayPreprocessor(
                    target_size=(224, 224),
                    apply_clahe=True,
                    clahe_clip_limit=clip,
                    clahe_tile_grid=grid,
                    normalize=False
                )
                result = prep.preprocess(sample_image)
                assert result.shape == (224, 224)


class TestPreprocessorConfigurations:
    """Tests para diferentes configuraciones del preprocesador."""
    
    @pytest.fixture
    def sample_image(self):
        np.random.seed(42)
        return np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    
    def test_no_clahe(self, sample_image):
        """Test sin CLAHE."""
        prep = XRayPreprocessor(apply_clahe=False, normalize=False)
        result = prep.preprocess(sample_image)
        assert result.shape == (224, 224)
    
    def test_no_normalization(self, sample_image):
        """Test sin normalización."""
        prep = XRayPreprocessor(apply_clahe=True, normalize=False)
        result = prep.preprocess(sample_image)
        assert result.dtype == np.uint8
    
    def test_custom_target_size(self, sample_image):
        """Test con tamaño personalizado."""
        prep = XRayPreprocessor(target_size=(128, 128))
        result = prep.preprocess(sample_image)
        assert result.shape == (128, 128)
