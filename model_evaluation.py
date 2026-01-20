"""
Script per la valutazione del modello ResNet50 addestrato sul dataset CUB-200-2011.
Carica il modello salvato e calcola le metriche di performance sul test set.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import (
    MODELS_DIR, 
    RESULTS_DIR, 
    BATCH_SIZE, 
    NUM_WORKERS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES
)
from src.model import create_model
from src.data_loader import CUB200Dataset, load_class_names


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Carica il modello ResNet50 salvato.
    
    Args:
        model_path: Path al file del modello (.pth)
        device: Device su cui caricare il modello
        
    Returns:
        Modello caricato e pronto per l'inference
    """
    print(f"Caricamento del modello da: {model_path}")
    
    # Crea l'architettura del modello
    model = create_model(
        model_type='resnet50',
        num_classes=NUM_CLASSES,
        pretrained=False,  # Non serve il pretrained perché carichiamo i pesi salvati
        device=device
    )
    
    # Carica i pesi salvati
    checkpoint = torch.load(model_path, map_location=device)
    
    # Se il checkpoint contiene un dizionario con 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"Modello addestrato per {checkpoint['epoch']} epoche")
        if 'best_acc' in checkpoint:
            print(f"Best accuracy durante il training: {checkpoint['best_acc']:.2f}%")
    else:
        # Altrimenti assume che il checkpoint sia direttamente lo state_dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Modello caricato con successo!\n")
    
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: dict
) -> dict:
    """
    Valuta il modello sul test set.
    
    Args:
        model: Modello da valutare
        test_loader: DataLoader per il test set
        device: Device per il calcolo
        class_names: Dizionario con i nomi delle classi
        
    Returns:
        Dizionario con le metriche di valutazione
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    print("Valutazione del modello...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calcola metriche globali
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average='weighted',
        zero_division=0
    )
    
    # Top-5 accuracy
    top5_acc = calculate_top_k_accuracy(all_probs, all_labels, k=5)
    
    print("\n" + "="*60)
    print("RISULTATI DELLA VALUTAZIONE")
    print("="*60)
    print(f"Accuracy:           {accuracy:.2f}%")
    print(f"Top-5 Accuracy:     {top5_acc:.2f}%")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1-Score (weighted):  {f1:.4f}")
    print("="*60 + "\n")
    
    # Metriche per classe
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average=None,
        zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'top5_accuracy': top5_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': conf_matrix,
        'per_class_metrics': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'support': support
        }
    }
    
    return results


def calculate_top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """
    Calcola la Top-K accuracy.
    
    Args:
        probs: Array di probabilità (N, num_classes)
        labels: Array di etichette vere (N,)
        k: Numero di predizioni top da considerare
        
    Returns:
        Top-K accuracy in percentuale
    """
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = 0
    for i, label in enumerate(labels):
        if label in top_k_preds[i]:
            correct += 1
    return (correct / len(labels)) * 100


def save_results(results: dict, class_names: dict, output_dir: Path):
    """
    Salva i risultati della valutazione.
    
    Args:
        results: Dizionario con i risultati
        class_names: Dizionario con i nomi delle classi
        output_dir: Directory dove salvare i risultati
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva metriche summary
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RISULTATI DELLA VALUTAZIONE DEL MODELLO\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy:              {results['accuracy']:.2f}%\n")
        f.write(f"Top-5 Accuracy:        {results['top5_accuracy']:.2f}%\n")
        f.write(f"Precision (weighted):  {results['precision']:.4f}\n")
        f.write(f"Recall (weighted):     {results['recall']:.4f}\n")
        f.write(f"F1-Score (weighted):   {results['f1_score']:.4f}\n")
    
    print(f"Summary salvato in: {summary_path}")
    
    # Salva metriche per classe
    per_class_df = pd.DataFrame({
        'class_id': range(NUM_CLASSES),
        'class_name': [class_names[i] for i in range(NUM_CLASSES)],
        'precision': results['per_class_metrics']['precision'],
        'recall': results['per_class_metrics']['recall'],
        'f1_score': results['per_class_metrics']['f1'],
        'support': results['per_class_metrics']['support']
    })
    per_class_df = per_class_df.sort_values('f1_score', ascending=False)
    
    per_class_path = output_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_path, index=False)
    print(f"Metriche per classe salvate in: {per_class_path}")
    
    # Mostra le migliori e peggiori 10 classi
    print("\n" + "="*60)
    print("TOP 10 CLASSI (per F1-score)")
    print("="*60)
    print(per_class_df[['class_name', 'f1_score', 'support']].head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("WORST 10 CLASSI (per F1-score)")
    print("="*60)
    print(per_class_df[['class_name', 'f1_score', 'support']].tail(10).to_string(index=False))
    print()


def plot_confusion_matrix(conf_matrix: np.ndarray, output_dir: Path, sample_size: int = 50):
    """
    Crea e salva una visualizzazione della confusion matrix.
    Per semplicità, mostra solo un subset di classi.
    
    Args:
        conf_matrix: Matrice di confusione
        output_dir: Directory dove salvare il plot
        sample_size: Numero di classi da visualizzare
    """
    # Visualizza solo un subset per leggibilità
    subset_matrix = conf_matrix[:sample_size, :sample_size]
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(subset_matrix, cmap='Blues', fmt='d', cbar=True)
    plt.title(f'Confusion Matrix (prime {sample_size} classi)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix salvata in: {output_path}")


def main():
    """Funzione principale per eseguire la valutazione del modello."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilizzato: {device}\n")
    
    # Path al modello
    model_path = MODELS_DIR / "best_resnet50_cub200.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato in: {model_path}")
    
    # Carica il modello
    model = load_model(str(model_path), device)
    
    # Prepara le trasformazioni per il test set
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Carica il test set
    print("Caricamento del test set...")
    test_dataset = CUB200Dataset(
        is_train=False,
        transform=test_transform,
        load_parts=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Test set: {len(test_dataset)} immagini\n")
    
    # Carica i nomi delle classi
    class_names = load_class_names()
    
    # Valuta il modello
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Salva i risultati
    print("\nSalvataggio dei risultati...")
    save_results(results, class_names, RESULTS_DIR)
    
    # Genera confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], RESULTS_DIR)
    
    print("\n" + "="*60)
    print("VALUTAZIONE COMPLETATA!")
    print("="*60)


if __name__ == "__main__":
    main()
