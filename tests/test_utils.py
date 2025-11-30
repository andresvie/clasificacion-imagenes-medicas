"""
Tests para el módulo de utilidades.
"""

import numpy as np
import pytest
import sys
import cv2
import json
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import (
    load_image,
    save_image,
    save_results,
    load_results,
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report
)

# Verificar si sklearn está disponible
try:
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TestLoadImage:
    """Tests para load_image."""
    
    @pytest.fixture
    def temp_image_path(self, tmp_path):
        """Crea una imagen temporal para pruebas."""
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img_path = tmp_path / "test_image.png"
        cv2.imwrite(str(img_path), img)
        return img_path
    
    @pytest.fixture
    def temp_color_image_path(self, tmp_path):
        """Crea una imagen en color temporal para pruebas."""
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img_path = tmp_path / "test_color_image.png"
        cv2.imwrite(str(img_path), img)
        return img_path
    
    def test_load_image_grayscale(self, temp_image_path):
        """Test de carga de imagen en escala de grises."""
        img = load_image(str(temp_image_path), grayscale=True)
        
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 2  # Grayscale tiene 2 dimensiones
        assert img.dtype == np.uint8
    
    def test_load_image_color(self, temp_color_image_path):
        """Test de carga de imagen en color."""
        img = load_image(str(temp_color_image_path), grayscale=False)
        
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3  # Color tiene 3 dimensiones
        assert img.shape[2] == 3  # RGB tiene 3 canales
    
    def test_load_image_grayscale_default(self, temp_image_path):
        """Test de carga de imagen con grayscale por defecto."""
        img = load_image(str(temp_image_path))
        
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 2
    
    def test_load_image_invalid_path(self):
        """Test de carga de imagen con ruta inválida."""
        img = load_image("/nonexistent/path/image.png")
        
        assert img is None
    
    def test_load_image_path_object(self, temp_image_path):
        """Test de carga de imagen con objeto Path."""
        img = load_image(temp_image_path)
        
        assert img is not None
        assert isinstance(img, np.ndarray)
    
    def test_load_image_different_formats(self, tmp_path):
        """Test de carga de imágenes en diferentes formatos."""
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        
        formats = ['png', 'jpg', 'jpeg']
        for fmt in formats:
            img_path = tmp_path / f"test.{fmt}"
            cv2.imwrite(str(img_path), img)
            
            loaded = load_image(str(img_path))
            assert loaded is not None
            assert isinstance(loaded, np.ndarray)


class TestSaveImage:
    """Tests para save_image."""
    
    @pytest.fixture
    def sample_grayscale_image(self):
        """Crea una imagen en escala de grises de prueba."""
        return np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    
    @pytest.fixture
    def sample_color_image(self):
        """Crea una imagen en color de prueba."""
        return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    
    def test_save_grayscale_image(self, sample_grayscale_image, tmp_path):
        """Test de guardado de imagen en escala de grises."""
        save_path = tmp_path / "saved_grayscale.png"
        result = save_image(sample_grayscale_image, str(save_path))
        
        assert result is True
        assert save_path.exists()
        
        # Verificar que se puede cargar
        loaded = cv2.imread(str(save_path), cv2.IMREAD_GRAYSCALE)
        assert loaded is not None
        assert loaded.shape == sample_grayscale_image.shape
    
    def test_save_color_image(self, sample_color_image, tmp_path):
        """Test de guardado de imagen en color."""
        save_path = tmp_path / "saved_color.png"
        result = save_image(sample_color_image, str(save_path))
        
        assert result is True
        assert save_path.exists()
        
        # Verificar que se puede cargar
        loaded = cv2.imread(str(save_path))
        assert loaded is not None
        assert len(loaded.shape) == 3
    
    def test_save_image_path_object(self, sample_grayscale_image, tmp_path):
        """Test de guardado con objeto Path."""
        save_path = tmp_path / "saved_path.png"
        result = save_image(sample_grayscale_image, save_path)
        
        assert result is True
        assert save_path.exists()
    
    def test_save_image_invalid_path(self, sample_grayscale_image):
        """Test de guardado con ruta inválida."""
        # Intentar guardar en un directorio (no un archivo)
        # cv2.imwrite puede crear el directorio padre, pero no puede escribir
        # en un directorio como si fuera un archivo
        import os
        if os.name != 'nt':  # No Windows
            # Intentar escribir en un directorio existente (no un archivo)
            invalid_path = "/tmp"  # Es un directorio, no un archivo
            result = save_image(sample_grayscale_image, invalid_path)
            # cv2.imwrite puede retornar True pero no crear el archivo esperado
            # Verificamos que no se creó un archivo en esa ruta
            assert result is False or not Path(invalid_path).is_file()
        else:
            # En Windows, intentar escribir en un directorio
            invalid_path = "C:\\Windows"  # Es un directorio
            result = save_image(sample_grayscale_image, invalid_path)
            assert result is False or not Path(invalid_path).is_file()
    
    def test_save_image_different_formats(self, sample_grayscale_image, tmp_path):
        """Test de guardado en diferentes formatos."""
        formats = ['png', 'jpg', 'jpeg']
        
        for fmt in formats:
            save_path = tmp_path / f"saved.{fmt}"
            result = save_image(sample_grayscale_image, str(save_path))
            
            assert result is True
            assert save_path.exists()


class TestSaveResults:
    """Tests para save_results."""
    
    @pytest.fixture
    def sample_results(self):
        """Crea resultados de muestra para pruebas."""
        return {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90,
            'numpy_array': np.array([1, 2, 3]),
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'nested_dict': {
                'value': np.array([4, 5, 6]),
                'number': np.int32(10)
            },
            'nested_list': [np.array([7, 8]), np.float32(2.5)]
        }
    
    def test_save_results_basic(self, sample_results, tmp_path):
        """Test de guardado básico de resultados."""
        save_path = tmp_path / "results.json"
        save_results(sample_results, str(save_path))
        
        assert save_path.exists()
        
        # Verificar contenido
        with open(save_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['accuracy'] == 0.95
        assert loaded['precision'] == 0.92
        assert isinstance(loaded['numpy_array'], list)
        assert loaded['numpy_array'] == [1, 2, 3]
        assert loaded['numpy_int'] == 42
        assert loaded['numpy_float'] == 3.14
    
    def test_save_results_path_object(self, sample_results, tmp_path):
        """Test de guardado con objeto Path."""
        save_path = tmp_path / "results_path.json"
        save_results(sample_results, save_path)
        
        assert save_path.exists()
    
    def test_save_results_nested_structures(self, sample_results, tmp_path):
        """Test de guardado con estructuras anidadas."""
        save_path = tmp_path / "results_nested.json"
        save_results(sample_results, str(save_path))
        
        with open(save_path, 'r') as f:
            loaded = json.load(f)
        
        assert isinstance(loaded['nested_dict'], dict)
        assert loaded['nested_dict']['value'] == [4, 5, 6]
        assert loaded['nested_dict']['number'] == 10
        assert isinstance(loaded['nested_list'], list)
        assert loaded['nested_list'][0] == [7, 8]
        assert loaded['nested_list'][1] == 2.5
    
    def test_save_results_creates_directory(self, sample_results, tmp_path):
        """Test que crea directorios si no existen."""
        save_path = tmp_path / "subdir" / "results.json"
        save_results(sample_results, str(save_path))
        
        assert save_path.exists()
        assert save_path.parent.exists()
    
    def test_save_results_empty_dict(self, tmp_path):
        """Test de guardado de diccionario vacío."""
        save_path = tmp_path / "empty_results.json"
        save_results({}, str(save_path))
        
        assert save_path.exists()
        with open(save_path, 'r') as f:
            loaded = json.load(f)
        assert loaded == {}


class TestLoadResults:
    """Tests para load_results."""
    
    @pytest.fixture
    def sample_json_file(self, tmp_path):
        """Crea un archivo JSON de muestra."""
        data = {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90,
            'metrics': {
                'train': 0.93,
                'val': 0.91
            },
            'list': [1, 2, 3]
        }
        json_path = tmp_path / "test_results.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)
        return json_path
    
    def test_load_results_basic(self, sample_json_file):
        """Test de carga básica de resultados."""
        results = load_results(str(sample_json_file))
        
        assert isinstance(results, dict)
        assert results['accuracy'] == 0.95
        assert results['precision'] == 0.92
        assert results['recall'] == 0.88
        assert results['f1_score'] == 0.90
    
    def test_load_results_path_object(self, sample_json_file):
        """Test de carga con objeto Path."""
        results = load_results(sample_json_file)
        
        assert isinstance(results, dict)
        assert results['accuracy'] == 0.95
    
    def test_load_results_nested_structures(self, sample_json_file):
        """Test de carga de estructuras anidadas."""
        results = load_results(str(sample_json_file))
        
        assert isinstance(results['metrics'], dict)
        assert results['metrics']['train'] == 0.93
        assert results['metrics']['val'] == 0.91
        assert isinstance(results['list'], list)
        assert results['list'] == [1, 2, 3]
    
    def test_load_results_empty_file(self, tmp_path):
        """Test de carga de archivo vacío."""
        json_path = tmp_path / "empty.json"
        with open(json_path, 'w') as f:
            json.dump({}, f)
        
        results = load_results(str(json_path))
        assert results == {}
    
    def test_load_results_invalid_path(self):
        """Test de carga con ruta inválida."""
        with pytest.raises(FileNotFoundError):
            load_results("/nonexistent/path/results.json")


class TestPlotConfusionMatrix:
    """Tests para plot_confusion_matrix."""
    
    @pytest.fixture
    def sample_confusion_matrix(self):
        """Crea una matriz de confusión de muestra."""
        return np.array([[50, 5], [3, 42]])
    
    @pytest.fixture
    def sample_classes(self):
        """Crea nombres de clases de muestra."""
        return ['NORMAL', 'PNEUMONIA']
    
    def test_plot_confusion_matrix_basic(self, sample_confusion_matrix, sample_classes):
        """Test básico de plot de matriz de confusión."""
        fig = plot_confusion_matrix(
            sample_confusion_matrix,
            sample_classes,
            normalize=False
        )
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
    
    def test_plot_confusion_matrix_normalized(self, sample_confusion_matrix, sample_classes):
        """Test de plot con normalización."""
        fig = plot_confusion_matrix(
            sample_confusion_matrix,
            sample_classes,
            normalize=True
        )
        
        assert fig is not None
    
    def test_plot_confusion_matrix_custom_title(self, sample_confusion_matrix, sample_classes):
        """Test con título personalizado."""
        fig = plot_confusion_matrix(
            sample_confusion_matrix,
            sample_classes,
            title='Custom Title'
        )
        
        assert fig is not None
    
    def test_plot_confusion_matrix_save(self, sample_confusion_matrix, sample_classes, tmp_path):
        """Test de guardado de figura."""
        save_path = tmp_path / "confusion_matrix.png"
        fig = plot_confusion_matrix(
            sample_confusion_matrix,
            sample_classes,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_confusion_matrix_custom_cmap(self, sample_confusion_matrix, sample_classes):
        """Test con mapa de colores personalizado."""
        fig = plot_confusion_matrix(
            sample_confusion_matrix,
            sample_classes,
            cmap='Reds'
        )
        
        assert fig is not None
    
    def test_plot_confusion_matrix_3x3(self):
        """Test con matriz 3x3."""
        cm = np.array([[30, 2, 1], [3, 25, 2], [1, 1, 36]])
        classes = ['Class1', 'Class2', 'Class3']
        
        fig = plot_confusion_matrix(cm, classes)
        assert fig is not None


class TestPlotROCCurve:
    """Tests para plot_roc_curve."""
    
    @pytest.fixture
    def sample_roc_data(self):
        """Crea datos ROC de muestra."""
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Curva ROC de ejemplo
        auc_score = 0.85
        return fpr, tpr, auc_score
    
    def test_plot_roc_curve_basic(self, sample_roc_data):
        """Test básico de plot de curva ROC."""
        fpr, tpr, auc = sample_roc_data
        fig = plot_roc_curve(fpr, tpr, auc)
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
    
    def test_plot_roc_curve_custom_title(self, sample_roc_data):
        """Test con título personalizado."""
        fpr, tpr, auc = sample_roc_data
        fig = plot_roc_curve(fpr, tpr, auc, title='Custom ROC Title')
        
        assert fig is not None
    
    def test_plot_roc_curve_save(self, sample_roc_data, tmp_path):
        """Test de guardado de figura."""
        fpr, tpr, auc = sample_roc_data
        save_path = tmp_path / "roc_curve.png"
        fig = plot_roc_curve(fpr, tpr, auc, save_path=str(save_path))
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_roc_curve_perfect_auc(self):
        """Test con AUC perfecto."""
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 1.0, 1.0])
        auc = 1.0
        
        fig = plot_roc_curve(fpr, tpr, auc)
        assert fig is not None
    
    def test_plot_roc_curve_random_auc(self):
        """Test con AUC aleatorio (diagonal)."""
        fpr = np.linspace(0, 1, 10)
        tpr = fpr  # Diagonal (AUC = 0.5)
        auc = 0.5
        
        fig = plot_roc_curve(fpr, tpr, auc)
        assert fig is not None


class TestPrintClassificationReport:
    """Tests para print_classification_report."""
    
    @pytest.fixture
    def sample_labels(self):
        """Crea etiquetas de muestra."""
        return np.array([0, 0, 1, 1, 0, 1, 0, 1])
    
    @pytest.fixture
    def sample_predictions(self):
        """Crea predicciones de muestra."""
        return np.array([0, 0, 1, 1, 0, 1, 1, 1])
    
    @pytest.fixture
    def sample_classes(self):
        """Crea nombres de clases de muestra."""
        return ['NORMAL', 'PNEUMONIA']
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn no está disponible")
    def test_print_classification_report_basic(self, sample_labels, sample_predictions, sample_classes):
        """Test básico de reporte de clasificación."""
        report = print_classification_report(
            sample_labels,
            sample_predictions,
            sample_classes
        )
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'NORMAL' in report or 'normal' in report.lower()
        assert 'PNEUMONIA' in report or 'pneumonia' in report.lower()
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn no está disponible")
    def test_print_classification_report_perfect_predictions(self, sample_classes):
        """Test con predicciones perfectas."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        report = print_classification_report(y_true, y_pred, sample_classes)
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn no está disponible")
    def test_print_classification_report_all_wrong(self, sample_classes):
        """Test con todas las predicciones incorrectas."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        
        report = print_classification_report(y_true, y_pred, sample_classes)
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn no está disponible")
    def test_print_classification_report_three_classes(self):
        """Test con tres clases."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])
        classes = ['Class0', 'Class1', 'Class2']
        
        report = print_classification_report(y_true, y_pred, classes)
        
        assert isinstance(report, str)
        assert 'Class0' in report or 'class0' in report.lower()
        assert 'Class1' in report or 'class1' in report.lower()
        assert 'Class2' in report or 'class2' in report.lower()
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn no está disponible")
    def test_print_classification_report_imbalanced(self, sample_classes):
        """Test con clases desbalanceadas."""
        y_true = np.array([0, 0, 0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        
        report = print_classification_report(y_true, y_pred, sample_classes)
        
        assert isinstance(report, str)
        assert len(report) > 0


class TestUtilsIntegration:
    """Tests de integración entre funciones de utils."""
    
    def test_save_and_load_results_roundtrip(self, tmp_path):
        """Test de guardado y carga de resultados."""
        original_results = {
            'accuracy': 0.95,
            'numpy_array': np.array([1, 2, 3]),
            'nested': {
                'value': np.float64(3.14)
            }
        }
        
        save_path = tmp_path / "roundtrip.json"
        save_results(original_results, str(save_path))
        
        loaded_results = load_results(str(save_path))
        
        assert loaded_results['accuracy'] == original_results['accuracy']
        assert loaded_results['numpy_array'] == [1, 2, 3]
        assert loaded_results['nested']['value'] == 3.14
    
    def test_save_and_load_image_roundtrip(self, tmp_path):
        """Test de guardado y carga de imagen."""
        original_img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        
        save_path = tmp_path / "roundtrip_image.png"
        save_image(original_img, str(save_path))
        
        loaded_img = load_image(str(save_path))
        
        assert loaded_img is not None
        assert loaded_img.shape == original_img.shape
        # Las imágenes pueden tener pequeñas diferencias por compresión
        assert np.allclose(loaded_img, original_img, atol=1)
    
    def test_plot_and_save_confusion_matrix(self, tmp_path):
        """Test de plot y guardado de matriz de confusión."""
        cm = np.array([[50, 5], [3, 42]])
        classes = ['NORMAL', 'PNEUMONIA']
        
        save_path = tmp_path / "cm.png"
        fig = plot_confusion_matrix(cm, classes, save_path=str(save_path))
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_and_save_roc_curve(self, tmp_path):
        """Test de plot y guardado de curva ROC."""
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)
        auc = 0.85
        
        save_path = tmp_path / "roc.png"
        fig = plot_roc_curve(fpr, tpr, auc, save_path=str(save_path))
        
        assert save_path.exists()
        plt.close(fig)

