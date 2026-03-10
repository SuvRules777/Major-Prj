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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import time
import pandas as pd
from pathlib import Path
import sys
from os.path import dirname, abspath

# Add parent directory to path for imports (kept for consistency, though
# we no longer import custom image-based CNNs in this script)
sys.path.insert(0, dirname(dirname(abspath(__file__))))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

"""Data loading and model comparison for fish biomass using local CSV data.

This version does not depend on Hugging Face datasets or image data.
Instead, it uses tabular measurements from the Fish Biomass dataset CSV.
"""


# ==================== DATA LOADING (LOCAL CSV) ====================

class TabularBiomassDataset(Dataset):
    """PyTorch Dataset for tabular fish measurements.

    Expects a pandas DataFrame and lists of feature and target columns.
    """

    def __init__(self, df: pd.DataFrame, feature_cols, target_col: str):
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(df[target_col].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def get_data_loaders(batch_size=32, num_workers=0):
    """Load local CSV data and prepare DataLoaders.

    Uses data/raw/fish_measurements.csv and splits it into
    train/test (80/20) for supervised regression.
    """

    print("=" * 60)
    print("LOADING DATASET FROM LOCAL CSV")
    print("=" * 60)

    # Resolve CSV path relative to project root
    project_root = Path(dirname(dirname(abspath(__file__))))
    csv_path = project_root / "data" / "raw" / "fish_measurements.csv"

    if not csv_path.exists():
        print(f"ERROR: CSV file not found at {csv_path}")
        return None, None

    df = pd.read_csv(csv_path)
    df = df.dropna()

    # Define feature and target columns
    feature_cols = ["Length1", "Length2", "Length3", "Height", "Width"]
    target_col = "Weight"

    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing expected columns in CSV: {missing_cols}")
        return None, None

    # Create full dataset
    full_dataset = TabularBiomassDataset(df, feature_cols, target_col)

    # Split the dataset into training and testing sets
    print("Splitting dataset into training and testing sets (80/20)...")
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, test_loader


TOP_LEVEL_INPUT_DIM = 5  # Length1, Length2, Length3, Height, Width


# ==================== MODEL DEFINITIONS (TABULAR) ====================

class LinearBiomassRegressor(nn.Module):
    """Single linear layer regressor for baseline comparison."""

    def __init__(self, input_dim: int = TOP_LEVEL_INPUT_DIM):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


class SmallMLPBiomassRegressor(nn.Module):
    """Two-layer MLP regressor."""

    def __init__(self, input_dim: int = TOP_LEVEL_INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class MediumMLPBiomassRegressor(nn.Module):
    """Three-layer MLP with hidden dimension 64."""

    def __init__(self, input_dim: int = TOP_LEVEL_INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLPBiomassRegressor(nn.Module):
    """Deeper MLP with dropout for regularization."""

    def __init__(self, input_dim: int = TOP_LEVEL_INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


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

    # Define models to compare (all operate on tabular measurements)
    model_configs = [
        ("Linear Regressor", LinearBiomassRegressor()),
        ("Small MLP Regressor", SmallMLPBiomassRegressor()),
        ("Medium MLP Regressor", MediumMLPBiomassRegressor()),
        ("Deep MLP Regressor", DeepMLPBiomassRegressor()),
    ]
    
    results = []
    histories = []
    model_names = []
    
    print("\n" + "=" * 60)
    print("BIOMASS ESTIMATION MODEL COMPARISON (Tabular CSV Dataset)")
    print("=" * 60)
    
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
