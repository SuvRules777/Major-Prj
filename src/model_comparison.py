"""
Model Comparison Script for Fish Biomass Estimation from Images
Compares multiple CNN architectures on the Fishnet.ai dataset from Hugging Face.

This script trains and evaluates different models for biomass estimation (regression),
using bounding box area as a proxy for biomass.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision import models

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import time
import pandas as pd
from pathlib import Path
from PIL import Image
from datasets import load_dataset
import sys
from os.path import dirname, abspath

# Add parent directory to path for imports
sys.path.insert(0, dirname(dirname(abspath(__file__))))

# Import custom model
from models.cnn_models import FishBiomassCNN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Image size for models
IMG_SIZE = 224  # Standard size for pretrained models


# ==================== DATA LOADING (Hugging Face) ====================

class HuggingFaceBiomassDataset(Dataset):
    """
    Custom PyTorch Dataset for the Hugging Face fishnet.ai dataset.
    It extracts images and calculates biomass from bounding box annotations.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image'].convert('RGB')
        
        # Calculate biomass proxy from bounding box area
        # Assumes the first object is the primary fish
        biomass = 0.0
        if item['objects']:
            bbox = item['objects']['bbox'][0]  # [xmin, ymin, xmax, ymax]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            biomass = width * height
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(biomass, dtype=torch.float32)


def get_data_loaders(batch_size=32, num_workers=0):
    """
    Load the fish dataset from Hugging Face and prepare DataLoaders.
    Trying multiple available fish datasets.
    """
    print("="*60)
    print("LOADING DATASET FROM HUGGING FACE")
    print("="*60)
    
    # Try multiple fish-related datasets
    dataset_options = [
        ("keremberke/fish-detection", "full"),
        ("Francesco/eurosat-fish", None),
        ("detection-datasets/fish", None)
    ]
    
    dataset = None
    for dataset_name, split_name in dataset_options:
        try:
            print(f"Trying to load '{dataset_name}'...")
            if split_name:
                dataset = load_dataset(dataset_name, split_name, split='train[:500]')
            else:
                dataset = load_dataset(dataset_name, split='train[:500]')
            print(f"Dataset '{dataset_name}' loaded successfully!")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    if dataset is None:
        print("\n" + "="*60)
        print("ERROR: Could not load any fish dataset from Hugging Face.")
        print("="*60)
        print("\nPlease ensure:")
        print("1. You have an internet connection")
        print("2. You have the 'datasets' library installed: pip install datasets")
        print("3. Try manually: from datasets import load_dataset")
        print("   load_dataset('keremberke/fish-detection', 'full')")
        return None, None

    # Define image transformations
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create the full dataset object first
    full_dataset = HuggingFaceBiomassDataset(dataset)

    # Split the dataset into training and testing sets
    print("Splitting dataset into training and testing sets (80/20)...")
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_subset, test_subset = random_split(list(range(len(full_dataset))), [train_size, test_size], 
                                             generator=torch.Generator().manual_seed(42))

    # Create PyTorch datasets with transforms
    train_dataset = HuggingFaceBiomassDataset(
        [full_dataset.hf_dataset[i] for i in train_subset.indices], 
        transform=train_transform
    )
    test_dataset = HuggingFaceBiomassDataset(
        [full_dataset.hf_dataset[i] for i in test_subset.indices], 
        transform=test_transform
    )

    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


# ==================== MODEL DEFINITIONS ====================

class SimpleBiomassCNN(nn.Module):
    """Simple CNN for biomass regression."""
    def __init__(self):
        super(SimpleBiomassCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

def get_pretrained_resnet18_regressor():
    """Pretrained ResNet18 adapted for biomass regression."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def get_pretrained_efficientnet_regressor():
    """Pretrained EfficientNet-B0 adapted for biomass regression."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


# ==================== TRAINING & EVALUATION ====================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """Train a regression model."""
    model.train()
    history = {'loss': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for images, biomass in pbar:
            images, biomass = images.to(device), biomass.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, biomass)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss/len(pbar):.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        history['loss'].append(epoch_loss)
        
    return history


def evaluate_model(model, test_loader):
    """Evaluate regression model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, biomass in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(biomass.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Avoid errors with zero variance in labels
    if len(np.unique(all_labels)) < 2:
        r2 = 0.0
    else:
        r2 = r2_score(all_labels, all_preds)

    metrics = {
        'mae': mean_absolute_error(all_labels, all_preds),
        'rmse': np.sqrt(mean_squared_error(all_labels, all_preds)),
        'r2_score': r2,
    }
    
    return metrics, all_preds, all_labels


# ==================== VISUALIZATION ====================

def plot_comparison_chart(results_df):
    """Bar chart comparing model regression metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = ['mae', 'rmse', 'r2_score']
    titles = ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R²)']
    
    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        bars = sns.barplot(x='model', y=metric, data=results_df, ax=ax, palette='viridis')
        ax.set_title(title)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars.patches:
            ax.annotate(f'{bar.get_height():.3f}',
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('biomass_model_comparison_metrics.png', dpi=150)
    plt.show()


def plot_predictions_vs_actual(labels, preds, model_name):
    """Scatter plot of predicted vs actual biomass."""
    plt.figure(figsize=(8, 8))
    plt.scatter(labels, preds, alpha=0.5)
    
    # Add a line for perfect correlation
    lims = [
        np.min([plt.xlim(), plt.ylim()]),
        np.max([plt.xlim(), plt.ylim()]),
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)
    
    plt.xlabel('Actual Biomass (bbox area)')
    plt.ylabel('Predicted Biomass (bbox area)')
    plt.title(f'Predicted vs Actual Biomass - {model_name}')
    plt.grid(True)
    plt.savefig(f'biomass_predictions_{model_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()


def plot_training_history(histories, model_names):
    """Plot training loss for all models."""
    plt.figure(figsize=(10, 6))
    for name, hist in zip(model_names, histories):
        plt.plot(hist['loss'], label=name, marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('biomass_training_curves.png', dpi=150)
    plt.show()


# ==================== MAIN EXECUTION ====================

def main():
    """Main function to run biomass model comparison."""
    
    # Hyperparameters
    BATCH_SIZE = 16  # Smaller batch size for larger images
    EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # Load data
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    if train_loader is None:
        print("Data loading failed. Exiting.")
        return

    # Define models to compare
    model_configs = [
        ('Custom FishBiomassCNN', FishBiomassCNN()),
        ('SimpleBiomassCNN', SimpleBiomassCNN()),
        ('ResNet18 Regressor', get_pretrained_resnet18_regressor()),
        ('EfficientNet-B0 Regressor', get_pretrained_efficientnet_regressor()),
    ]
    
    results = []
    histories = []
    model_names = []
    
    print("\n" + "="*60)
    print("BIOMASS ESTIMATION MODEL COMPARISON (Hugging Face Dataset)")
    print("="*60)
    
    for name, model in model_configs:
        print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")
        
        model = model.to(device)
        num_params = count_parameters(model)
        print(f"Parameters: {num_params:,}")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        start_time = time.time()
        history = train_model(model, train_loader, criterion, optimizer, EPOCHS)
        train_time = time.time() - start_time
        
        histories.append(history)
        model_names.append(name)
        
        metrics, preds, labels = evaluate_model(model, test_loader)
        
        results.append({
            'model': name,
            'parameters': num_params,
            'train_time': train_time,
            **metrics
        })
        
        print(f"\n{name} Results:")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  R² Score: {metrics['r2_score']:.3f}")
        print(f"  Training Time: {train_time:.2f}s")
        
        if 'ResNet' in name:
            plot_predictions_vs_actual(labels, preds, name)
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    results_df.to_csv('biomass_model_comparison_results.csv', index=False)
    print("\nSaved: biomass_model_comparison_results.csv")
    
    print("\nGenerating visualizations...")
    plot_comparison_chart(results_df)
    plot_training_history(histories, model_names)
    
    print("\n" + "="*60)
    print("FINAL RANKING (by R² Score)")
    print("="*60)
    ranked = results_df.sort_values('r2_score', ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"{i}. {row['model']}: R²={row['r2_score']:.3f}, "
              f"RMSE={row['rmse']:.3f}, {row['parameters']:,} params")
    
    return results_df


if __name__ == '__main__':
    results = main()
