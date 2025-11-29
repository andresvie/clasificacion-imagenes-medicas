"""
Script principal para ejecutar el pipeline completo de clasificación de neumonía.

Uso:
    python src/main.py --data_root ./data/chest_xray
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

from data_loader import DataLoader
from preprocessing import XRayPreprocessor
from feature_extraction import FeatureExtractor
from utils import save_results, plot_confusion_matrix, plot_roc_curve


def main(args):
    """Pipeline principal de clasificación."""
    
    print("=" * 70)
    print("CLASIFICACIÓN DE NEUMONÍA EN RADIOGRAFÍAS DE TÓRAX")
    print("=" * 70)
    
    # Crear directorio de resultados
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos...")
    loader = DataLoader(args.data_root)
    counts = loader.get_class_counts()
    print(f"  Train: {counts.get('train', {})}")
    print(f"  Val: {counts.get('val', {})}")
    print(f"  Test: {counts.get('test', {})}")
    
    # 2. Inicializar preprocesador y extractor
    print("\n[2/5] Inicializando pipeline...")
    preprocessor = XRayPreprocessor(
        target_size=(224, 224),
        apply_clahe=True,
        clahe_clip_limit=2.0,
        segment_lungs=False,
        normalize=True
    )
    
    extractor = FeatureExtractor(
        include_intensity=True,
        include_glcm=True,
        include_lbp=True,
        include_gabor=True,
        include_gradient=True
    )
    
    # 3. Procesar imágenes y extraer características
    print("\n[3/5] Procesando imágenes y extrayendo características...")
    
    def process_split(split_name, max_samples=None):
        images, labels = loader.load_split(split_name, max_per_class=max_samples)
        features_list = []
        
        for img in tqdm(images, desc=f"  {split_name}"):
            img_processed = preprocessor.preprocess(img)
            features = extractor.extract_all_features(img_processed)
            features_list.append(list(features.values()))
        
        return np.array(features_list), labels
    
    # Procesar train y test
    max_samples = args.max_samples if args.max_samples > 0 else None
    X_train, y_train = process_split('train', max_samples)
    X_test, y_test = process_split('test', max_samples)
    
    print(f"\n  Train: {X_train.shape[0]} muestras, {X_train.shape[1]} características")
    print(f"  Test: {X_test.shape[0]} muestras, {X_test.shape[1]} características")
    
    # 4. Entrenar y evaluar modelos
    print("\n[4/5] Entrenando modelos...")
    
    # Normalizar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    print("\n  Entrenando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    
    # SVM
    print("  Entrenando SVM...")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]
    
    # 5. Evaluar y guardar resultados
    print("\n[5/5] Evaluando y guardando resultados...")
    
    classes = ['NORMAL', 'PNEUMONIA']
    
    # Métricas Random Forest
    print("\n" + "=" * 50)
    print("RANDOM FOREST")
    print("=" * 50)
    print(classification_report(y_test, y_pred_rf, target_names=classes))
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    print(f"AUC-ROC: {auc_rf:.4f}")
    
    # Métricas SVM
    print("\n" + "=" * 50)
    print("SVM")
    print("=" * 50)
    print(classification_report(y_test, y_pred_svm, target_names=classes))
    auc_svm = roc_auc_score(y_test, y_prob_svm)
    print(f"AUC-ROC: {auc_svm:.4f}")
    
    # Guardar resultados
    results = {
        'random_forest': {
            'accuracy': float(np.mean(y_pred_rf == y_test)),
            'auc_roc': float(auc_rf),
            'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist()
        },
        'svm': {
            'accuracy': float(np.mean(y_pred_svm == y_test)),
            'auc_roc': float(auc_svm),
            'confusion_matrix': confusion_matrix(y_test, y_pred_svm).tolist()
        }
    }
    
    save_results(results, results_dir / 'metrics.json')
    
    # Guardar visualizaciones
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plot_confusion_matrix(cm_rf, classes, title='Random Forest - Confusion Matrix',
                         save_path=results_dir / 'confusion_matrix_rf.png')
    
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    plot_confusion_matrix(cm_svm, classes, title='SVM - Confusion Matrix',
                         save_path=results_dir / 'confusion_matrix_svm.png')
    
    print(f"\n✓ Resultados guardados en: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clasificación de neumonía en radiografías')
    parser.add_argument('--data_root', type=str, default='./data/chest-xray-pneumonia/chest_xray',
                       help='Ruta al dataset')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directorio para guardar resultados')
    parser.add_argument('--max_samples', type=int, default=0,
                       help='Máximo de muestras por clase (0 = sin límite)')
    
    args = parser.parse_args()
    main(args)
