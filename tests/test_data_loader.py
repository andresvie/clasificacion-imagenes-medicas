"""
Tests para el módulo de carga de datos.
"""

import numpy as np
import pytest
import sys
import cv2
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader


class TestDataLoader:
    """Tests para DataLoader."""
    
    @pytest.fixture
    def temp_dataset(self, tmp_path):
        """Crea una estructura de dataset temporal para pruebas."""
        # Crear estructura de directorios
        splits = ['train', 'val', 'test']
        classes = ['NORMAL', 'PNEUMONIA']
        
        # Crear imágenes de prueba
        test_images = {}
        
        for split in splits:
            split_path = tmp_path / split
            for class_name in classes:
                class_path = split_path / class_name
                class_path.mkdir(parents=True, exist_ok=True)
                
                # Crear algunas imágenes de prueba
                images = []
                for i in range(3):
                    # Crear imagen de prueba (512x512 grayscale)
                    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
                    
                    # Guardar en diferentes formatos
                    if i == 0:
                        img_path = class_path / f"image_{i}.jpeg"
                        cv2.imwrite(str(img_path), img)
                    elif i == 1:
                        img_path = class_path / f"image_{i}.jpg"
                        cv2.imwrite(str(img_path), img)
                    else:
                        img_path = class_path / f"image_{i}.png"
                        cv2.imwrite(str(img_path), img)
                    
                    images.append(img_path)
                
                test_images[f"{split}_{class_name}"] = images
        
        return tmp_path, test_images
    
    @pytest.fixture
    def data_loader(self, temp_dataset):
        """Crea un DataLoader con el dataset temporal."""
        data_root, _ = temp_dataset
        return DataLoader(str(data_root))
    
    def test_initialization(self, temp_dataset):
        """Test de inicialización del DataLoader."""
        data_root, _ = temp_dataset
        loader = DataLoader(str(data_root))
        
        assert loader.data_root == Path(data_root)
        assert isinstance(loader.image_paths, dict)
        assert 'train' in loader.image_paths
        assert 'val' in loader.image_paths
        assert 'test' in loader.image_paths
    
    def test_scan_dataset(self, data_loader):
        """Test del escaneo del dataset."""
        image_paths = data_loader.image_paths
        
        # Verificar estructura
        for split in ['train', 'val', 'test']:
            assert split in image_paths
            assert 'NORMAL' in image_paths[split]
            assert 'PNEUMONIA' in image_paths[split]
            # Debe haber 3 imágenes por clase (una en cada formato)
            assert len(image_paths[split]['NORMAL']) == 3
            assert len(image_paths[split]['PNEUMONIA']) == 3
    
    def test_get_image_paths(self, data_loader):
        """Test de obtención de rutas de imágenes."""
        paths = data_loader.get_image_paths()
        
        assert isinstance(paths, dict)
        assert paths == data_loader.image_paths
        
        # Verificar que las rutas son Path objects
        for split_paths in paths.values():
            for class_paths in split_paths.values():
                for path in class_paths:
                    assert isinstance(path, Path)
    
    def test_get_class_counts(self, data_loader):
        """Test de conteo de clases."""
        counts = data_loader.get_class_counts()
        
        assert isinstance(counts, dict)
        for split in ['train', 'val', 'test']:
            assert split in counts
            assert 'NORMAL' in counts[split]
            assert 'PNEUMONIA' in counts[split]
            assert counts[split]['NORMAL'] == 3
            assert counts[split]['PNEUMONIA'] == 3
    
    def test_load_image_grayscale(self, data_loader, temp_dataset):
        """Test de carga de imagen en escala de grises."""
        _, test_images = temp_dataset
        test_path = test_images['train_NORMAL'][0]
        
        img = data_loader.load_image(test_path, grayscale=True)
        
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 2  # Grayscale tiene 2 dimensiones
        assert img.dtype == np.uint8
    
    def test_load_image_color(self, data_loader, temp_dataset):
        """Test de carga de imagen en color."""
        _, test_images = temp_dataset
        test_path = test_images['train_NORMAL'][0]
        
        img = data_loader.load_image(test_path, grayscale=False)
        
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3  # Color tiene 3 dimensiones
        assert img.shape[2] == 3  # RGB tiene 3 canales
    
    def test_load_image_invalid_path(self, data_loader):
        """Test de carga de imagen con ruta inválida."""
        invalid_path = Path('/nonexistent/path/image.jpg')
        img = data_loader.load_image(invalid_path)
        
        assert img is None
    
    def test_load_image_different_formats(self, data_loader, temp_dataset):
        """Test de carga de imágenes en diferentes formatos."""
        _, test_images = temp_dataset
        
        # Probar todos los formatos
        for format_type in ['jpeg', 'jpg', 'png']:
            if format_type == 'jpeg':
                test_path = test_images['train_NORMAL'][0]
            elif format_type == 'jpg':
                test_path = test_images['train_NORMAL'][1]
            else:
                test_path = test_images['train_NORMAL'][2]
            
            img = data_loader.load_image(test_path)
            assert img is not None
            assert isinstance(img, np.ndarray)
    
    def test_load_split(self, data_loader):
        """Test de carga de un split completo."""
        images, labels = data_loader.load_split('train')
        
        assert isinstance(images, list)
        assert isinstance(labels, np.ndarray)
        assert len(images) == 6  # 3 NORMAL + 3 PNEUMONIA
        assert len(labels) == 6
        
        # Verificar etiquetas
        assert labels[0] == 0  # NORMAL
        assert labels[3] == 1  # PNEUMONIA
        
        # Verificar que todas las imágenes se cargaron correctamente
        for img in images:
            assert img is not None
            assert isinstance(img, np.ndarray)
    
    def test_load_split_with_max_per_class(self, data_loader):
        """Test de carga de split con límite por clase."""
        images, labels = data_loader.load_split('train', max_per_class=2)
        
        assert len(images) == 4  # 2 NORMAL + 2 PNEUMONIA
        assert len(labels) == 4
    
    def test_load_split_empty(self, tmp_path):
        """Test de carga de split vacío."""
        # Crear estructura vacía
        (tmp_path / 'train' / 'NORMAL').mkdir(parents=True)
        (tmp_path / 'train' / 'PNEUMONIA').mkdir(parents=True)
        
        loader = DataLoader(str(tmp_path))
        images, labels = loader.load_split('train')
        
        assert len(images) == 0
        assert len(labels) == 0
    
    def test_get_sample_images(self, data_loader):
        """Test de obtención de imágenes de muestra."""
        samples = data_loader.get_sample_images('train', n_per_class=2)
        
        assert isinstance(samples, dict)
        assert 'NORMAL' in samples
        assert 'PNEUMONIA' in samples
        assert len(samples['NORMAL']) == 2
        assert len(samples['PNEUMONIA']) == 2
        
        # Verificar que las imágenes se cargaron
        for class_name in ['NORMAL', 'PNEUMONIA']:
            for img in samples[class_name]:
                assert img is not None
                assert isinstance(img, np.ndarray)
    
    def test_get_sample_images_more_than_available(self, data_loader):
        """Test cuando se piden más muestras de las disponibles."""
        samples = data_loader.get_sample_images('train', n_per_class=10)
        
        # Debe retornar solo las disponibles (3)
        assert len(samples['NORMAL']) == 3
        assert len(samples['PNEUMONIA']) == 3
    
    def test_analyze_dimensions(self, data_loader):
        """Test de análisis de dimensiones."""
        stats = data_loader.analyze_dimensions(n_samples=2)
        
        assert isinstance(stats, dict)
        assert 'mean_height' in stats
        assert 'mean_width' in stats
        assert 'std_height' in stats
        assert 'std_width' in stats
        assert 'min_height' in stats
        assert 'max_height' in stats
        assert 'min_width' in stats
        assert 'max_width' in stats
        
        # Verificar que los valores son razonables
        assert stats['mean_height'] > 0
        assert stats['mean_width'] > 0
        assert stats['min_height'] <= stats['mean_height'] <= stats['max_height']
        assert stats['min_width'] <= stats['mean_width'] <= stats['max_width']
    
    def test_analyze_dimensions_empty_dataset(self, tmp_path):
        """Test de análisis con dataset vacío."""
        # Crear estructura vacía
        for split in ['train', 'val', 'test']:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                (tmp_path / split / class_name).mkdir(parents=True)
        
        loader = DataLoader(str(tmp_path))
        
        # Debe manejar dataset vacío sin error
        # (puede lanzar error si no hay imágenes, dependiendo de la implementación)
        with pytest.raises((ValueError, ZeroDivisionError)):
            stats = loader.analyze_dimensions()


class TestDataLoaderEdgeCases:
    """Tests para casos límite y errores."""
    
    def test_missing_split_directory(self, tmp_path):
        """Test cuando falta un directorio de split."""
        # Crear solo train y val, sin test
        (tmp_path / 'train' / 'NORMAL').mkdir(parents=True)
        (tmp_path / 'train' / 'PNEUMONIA').mkdir(parents=True)
        (tmp_path / 'val' / 'NORMAL').mkdir(parents=True)
        (tmp_path / 'val' / 'PNEUMONIA').mkdir(parents=True)
        
        loader = DataLoader(str(tmp_path))
        
        # Debe manejar el split faltante
        assert 'train' in loader.image_paths
        assert 'val' in loader.image_paths
        # test puede no estar si no existe el directorio
    
    def test_missing_class_directory(self, tmp_path):
        """Test cuando falta un directorio de clase."""
        # Crear solo NORMAL, sin PNEUMONIA
        (tmp_path / 'train' / 'NORMAL').mkdir(parents=True)
        
        loader = DataLoader(str(tmp_path))
        
        # Debe manejar la clase faltante
        assert 'NORMAL' in loader.image_paths['train']
        assert 'PNEUMONIA' in loader.image_paths['train']
        assert len(loader.image_paths['train']['PNEUMONIA']) == 0
    
    def test_nonexistent_data_root(self):
        """Test con ruta raíz inexistente."""
        loader = DataLoader('/nonexistent/path')
        
        # Debe inicializar sin error pero sin imágenes
        assert isinstance(loader.image_paths, dict)
        assert len(loader.image_paths) == 0 or all(
            len(paths) == 0 
            for split_paths in loader.image_paths.values() 
            for paths in split_paths.values()
        )
    
    def test_load_split_invalid_split(self, tmp_path):
        """Test de carga de split inválido."""
        (tmp_path / 'train' / 'NORMAL').mkdir(parents=True)
        
        loader = DataLoader(str(tmp_path))
        
        # Debe manejar split inválido
        with pytest.raises(KeyError):
            images, labels = loader.load_split('invalid_split')
    
    def test_get_sample_images_invalid_split(self, tmp_path):
        """Test de muestras con split inválido."""
        (tmp_path / 'train' / 'NORMAL').mkdir(parents=True)
        
        loader = DataLoader(str(tmp_path))
        
        # Debe manejar split inválido
        with pytest.raises(KeyError):
            samples = loader.get_sample_images('invalid_split')


class TestDataLoaderImageFormats:
    """Tests específicos para diferentes formatos de imagen."""
    
    @pytest.fixture
    def format_dataset(self, tmp_path):
        """Crea dataset con diferentes formatos de imagen."""
        split_path = tmp_path / 'train' / 'NORMAL'
        split_path.mkdir(parents=True)
        
        # Crear imagen de prueba
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        
        # Guardar en diferentes formatos
        formats = {
            'jpeg': split_path / 'test.jpeg',
            'jpg': split_path / 'test.jpg',
            'png': split_path / 'test.png'
        }
        
        for path in formats.values():
            cv2.imwrite(str(path), img)
        
        return tmp_path, formats
    
    def test_all_formats_detected(self, format_dataset):
        """Test que todos los formatos son detectados."""
        data_root, _ = format_dataset
        loader = DataLoader(str(data_root))
        
        paths = loader.image_paths['train']['NORMAL']
        assert len(paths) == 3  # Debe encontrar los 3 formatos
    
    def test_load_different_formats(self, format_dataset):
        """Test de carga de diferentes formatos."""
        data_root, formats = format_dataset
        loader = DataLoader(str(data_root))
        
        for format_path in formats.values():
            img = loader.load_image(format_path)
            assert img is not None
            assert img.shape == (256, 256)


class TestDataLoaderConstants:
    """Tests para constantes de la clase."""
    
    def test_classes_constant(self):
        """Test de constante CLASSES."""
        assert DataLoader.CLASSES == ['NORMAL', 'PNEUMONIA']
        assert len(DataLoader.CLASSES) == 2
    
    def test_splits_constant(self):
        """Test de constante SPLITS."""
        assert DataLoader.SPLITS == ['train', 'val', 'test']
        assert len(DataLoader.SPLITS) == 3

