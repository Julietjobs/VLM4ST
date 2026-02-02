"""
Data loader for UniST dataset
"""
import numpy as np
import torch
import torch.nn.functional as F
import json
import datetime
import random
from torch.utils.data import Dataset, DataLoader


class MinMaxNormalization(object):
    """
    MinMax Normalization: Scale data to [-1, 1] range
    
    Formula:
        1. x = (x - min) / (max - min)  -> [0, 1]
        2. x = x * 2 - 1                -> [-1, 1]
    
    Methods:
        fit: Compute min and max from training data
        transform: Apply normalization using fitted min/max
        fit_transform: Fit and transform in one step
        inverse_transform: Convert normalized data back to original scale
    """

    def __init__(self):
        self._min = None
        self._max = None

    def fit(self, X):
        """Fit the scaler by computing min and max values"""
        self._min = X.min()
        self._max = X.max()
        print(f"MinMaxNormalization fitted - min: {self._min:.4f}, max: {self._max:.4f}")

    def transform(self, X):
        """Transform data to [-1, 1] range using fitted min/max"""
        if self._min is None or self._max is None:
            raise ValueError("Scaler must be fitted before transform")
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Convert normalized data back to original scale"""
        if self._min is None or self._max is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


class UniSTDataset(Dataset):
    """
    PyTorch Dataset for UniST spatio-temporal data
    
    Returns only spatial_temporal data and timestamps (ignores periodical data)
    This matches the VLM4ST model input requirements.
    
    Args:
        spatial_temporal_data: torch.Tensor of shape (N, T, H, W)
        timestamps: torch.Tensor of shape (N, T, 2) where [:,:,0] is weekday, [:,:,1] is time_of_day
    
    Returns:
        Dictionary with keys:
            'spatial_temporal': (T, H, W) - spatio-temporal observation
            'timestamps': (T, 2) - corresponding time information
    """
    
    def __init__(self, spatial_temporal_data, timestamps):
        self.spatial_temporal = spatial_temporal_data
        self.timestamps = timestamps
        
        assert len(self.spatial_temporal) == len(self.timestamps), \
            "Spatial-temporal data and timestamps must have same number of samples"
    
    def __len__(self):
        return len(self.spatial_temporal)
    
    def __getitem__(self, idx):
        return {
            'spatial_temporal': self.spatial_temporal[idx],  # (T, H, W)
            'timestamps': self.timestamps[idx]               # (T, 2)
        }



def resize_spatiotemporal(X, target_H, target_W):
    """
    Resize spatiotemporal data from (N, T, H, W) to (N, T, target_H, target_W)
    
    Args:
        X: torch.Tensor of shape (N, T, H, W)
        target_H: target height
        target_W: target width
    
    Returns:
        Resized tensor of shape (N, T, target_H, target_W)
    """
    N, T, H, W = X.shape
    
    # Reshape to (N*T, 1, H, W) for interpolation
    X_reshaped = X.view(N * T, 1, H, W)
    
    # Use bilinear interpolation (smooth, good for continuous spatial data)
    X_resized = F.interpolate(
        X_reshaped, 
        size=(target_H, target_W), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Reshape back to (N, T, target_H, target_W)
    X_resized = X_resized.view(N, T, target_H, target_W)
    
    return X_resized


def load_single_dataset(json_path, target_H=None, target_W=None, batch_size=32, normalize=True, num_workers=4):
    """
    Load a single UniST dataset from JSON file and create DataLoaders
    
    Args:
        json_path: Path to the JSON dataset file (e.g., 'Dataset/BikeNYC_short.json')
        target_H: Target height (None = use original, otherwise resize to this)
        target_W: Target width (None = use original, otherwise resize to this)
        batch_size: Base batch size for training (will be adjusted based on spatial size)
        normalize: Whether to apply MinMax normalization to [-1, 1]
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        scaler: Fitted MinMaxNormalization scaler (for inverse transform)
        data_info: Dictionary containing dataset information (N, T, H, W, dataset_name)
    """
    
    print(f"\n{'='*80}")
    print(f"Loading dataset from: {json_path}")
    print(f"{'='*80}")
    
    # Extract dataset name from path
    dataset_name = json_path.split('/')[-1].replace('_short.json', '')
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data_all = json.load(f)
    
    # ==================== Load Spatio-temporal Data ====================
    # Shape: (N, T, H, W) where N=samples, T=timesteps, H=height, W=width
    # Note: We only use X[0] (spatio-temporal), ignore X[1] (periodical data)
    X_train = torch.tensor(data_all['X_train'][0], dtype=torch.float32)
    X_test = torch.tensor(data_all['X_test'][0], dtype=torch.float32)
    X_val = torch.tensor(data_all['X_val'][0], dtype=torch.float32)
    
    N_train, T, H_orig, W_orig = X_train.shape
    print(f"Spatio-temporal data loaded:")
    print(f"  Train: {X_train.shape} (N={N_train}, T={T}, H={H_orig}, W={W_orig})")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # ==================== Resize if needed ====================
    if target_H is not None and target_W is not None:
        if H_orig != target_H or W_orig != target_W:
            print(f"Resizing from ({H_orig}, {W_orig}) to ({target_H}, {target_W})")
            X_train = resize_spatiotemporal(X_train, target_H, target_W)
            X_test = resize_spatiotemporal(X_test, target_H, target_W)
            X_val = resize_spatiotemporal(X_val, target_H, target_W)
            H, W = target_H, target_W
        else:
            H, W = H_orig, W_orig
            print(f"Dimensions match target, no resizing needed")
    else:
        H, W = H_orig, W_orig
        print(f"No target dimensions specified, using original size")
    
    # ==================== Process Timestamps ====================
    # Convert timestamps to (N, T, 2) format where:
    #   [:, :, 0] = weekday (0=Monday, ..., 6=Sunday)
    #   [:, :, 1] = time_of_day (0-47 for 30-minute intervals)
    
    # Different datasets have different timestamp formats, process accordingly:
    
    if 'TaxiBJ' in dataset_name:
        # TaxiBJ: timestamps are datetime strings like '2015-01-01 08:00:00'
        print(f"Processing TaxiBJ timestamps (datetime string format)...")
        X_train_ts = data_all['timestamps']['train']
        X_test_ts = data_all['timestamps']['test']
        X_val_ts = data_all['timestamps']['val']
        
        # Parse datetime strings and extract weekday + time_of_day
        X_train_ts = torch.tensor([
            [(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),
              datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2 + 
              int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) 
             for i in t] for t in X_train_ts
        ])
        X_test_ts = torch.tensor([
            [(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),
              datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2 + 
              int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) 
             for i in t] for t in X_test_ts
        ])
        X_val_ts = torch.tensor([
            [(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),
              datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2 + 
              int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) 
             for i in t] for t in X_val_ts
        ])
    
    elif 'Crowd' in dataset_name or 'Cellular' in dataset_name or 'Traffic_log' in dataset_name:
        # Crowd/Cellular: timestamps are integer indices
        print(f"Processing Crowd/Cellular timestamps (integer index format)...")
        X_train_ts = data_all['timestamps']['train']
        X_test_ts = data_all['timestamps']['test']
        X_val_ts = data_all['timestamps']['val']
        
        # Convert integer indices to weekday and time_of_day
        # Formula: weekday = ((i % (24*2*7)) // (24*2) + 2) % 7, time_of_day = i % (24*2)
        X_train_ts = torch.tensor([
            [((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_train_ts
        ])
        X_test_ts = torch.tensor([
            [((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_test_ts
        ])
        X_val_ts = torch.tensor([
            [((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_val_ts
        ])
    
    elif ('TaxiNYC' in dataset_name or 'BikeNYC' in dataset_name or 'TDrive' in dataset_name or 
          'Traffic' in dataset_name or 'DC' in dataset_name or 'Austin' in dataset_name or 
          'Porto' in dataset_name or 'CHI' in dataset_name or 'METR-LA' in dataset_name or 
          'CrowdBJ' in dataset_name):
        # These datasets: timestamps already in [weekday, time_of_day] format
        print(f"Processing {dataset_name} timestamps (already in correct format)...")
        X_train_ts = torch.tensor(data_all['timestamps']['train'], dtype=torch.long)
        X_test_ts = torch.tensor(data_all['timestamps']['test'], dtype=torch.long)
        X_val_ts = torch.tensor(data_all['timestamps']['val'], dtype=torch.long)
    
    else:
        # Default: assume already in correct format
        print(f"Processing timestamps (assuming correct format)...")
        X_train_ts = torch.tensor(data_all['timestamps']['train'], dtype=torch.long)
        X_test_ts = torch.tensor(data_all['timestamps']['test'], dtype=torch.long)
        X_val_ts = torch.tensor(data_all['timestamps']['val'], dtype=torch.long)
    
    print(f"Timestamps processed: shape {X_train_ts.shape}")
    
    # ==================== Normalization ====================
    if normalize:
        my_scaler = MinMaxNormalization()
        # Fit scaler on global min/max across all splits
        MAX = max(torch.max(X_train).item(), torch.max(X_test).item(), torch.max(X_val).item())
        MIN = min(torch.min(X_train).item(), torch.min(X_test).item(), torch.min(X_val).item())
        my_scaler.fit(np.array([MIN, MAX]))
        
        # Apply normalization
        X_train = torch.from_numpy(my_scaler.transform(X_train.numpy().reshape(-1,1)).reshape(X_train.shape)).float()
        X_test = torch.from_numpy(my_scaler.transform(X_test.numpy().reshape(-1,1)).reshape(X_test.shape)).float()
        X_val = torch.from_numpy(my_scaler.transform(X_val.numpy().reshape(-1,1)).reshape(X_val.shape)).float()
    else:
        my_scaler = None
    
    # ==================== Create Datasets ====================
    train_dataset = UniSTDataset(X_train, X_train_ts)
    val_dataset = UniSTDataset(X_val, X_val_ts)
    test_dataset = UniSTDataset(X_test, X_test_ts)
    
    # ==================== Adaptive Batch Size ====================
    # Adjust batch size based on spatial dimensions to avoid OOM
    if H + W < 32:
        train_bs = batch_size
    elif H + W < 48:
        train_bs = max(1, batch_size // 2)
    elif H + W < 64:
        train_bs = max(1, batch_size // 4)
    else:
        train_bs = max(1, batch_size // 8)
    
    print(f"\nBatch size adapted based on spatial size (H+W={H+W}):")
    print(f"  Train batch size: {train_bs}")
    print(f"  Val/Test batch size: {train_bs * 4}")
    
    # ==================== Create DataLoaders ====================
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_bs * 4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_bs * 4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # ==================== Data Info ====================
    data_info = {
        'dataset_name': dataset_name,
        'N_train': N_train,
        'N_val': len(val_dataset),
        'N_test': len(test_dataset),
        'T': T,
        'H': H,
        'W': W,
        'H_original': H_orig,
        'W_original': W_orig,
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader)
    }
    
    print(f"\nDataLoaders created successfully!")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"{'='*80}\n")
    
    return train_loader, val_loader, test_loader, my_scaler, data_info

def load_multiple_datasets(json_paths, target_H, target_W, batch_size=32, normalize=True, num_workers=4):
    """
    Load multiple UniST datasets and create combined DataLoaders
    
    All datasets are resized to unified dimensions (target_H, target_W) for compatibility.
    Training data from all datasets are combined, while validation and test remain separate
    per dataset for proper evaluation.
    
    Args:
        json_paths: List of paths to JSON dataset files
        target_H: Unified target height (all datasets resized to this)
        target_W: Unified target width (all datasets resized to this)
        batch_size: Base batch size for training
        normalize: Whether to apply MinMax normalization
        num_workers: Number of worker processes
    
    Returns:
        train_loader: Combined DataLoader with data from all datasets (shuffled)
        val_loaders: List of validation DataLoaders (one per dataset)
        test_loaders: List of test DataLoaders (one per dataset)
        scalers: Dictionary mapping dataset_name to its scaler
        data_infos: Dictionary mapping dataset_name to its data_info
    """
    
    print(f"\n{'='*80}")
    print(f"Loading Multiple Datasets: {len(json_paths)} datasets")
    print(f"Target unified dimensions: H={target_H}, W={target_W}")
    print(f"{'='*80}")
    
    train_datasets = []
    val_loaders = []
    test_loaders = []
    scalers = {}
    data_infos = {}
    
    # Load each dataset with unified dimensions
    for json_path in json_paths:
        train_loader, val_loader, test_loader, scaler, data_info = load_single_dataset(
            json_path, target_H=target_H, target_W=target_W, 
            batch_size=batch_size, normalize=normalize, num_workers=num_workers
        )
        
        dataset_name = data_info['dataset_name']
        train_datasets.append(train_loader.dataset)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
        scalers[dataset_name] = scaler
        data_infos[dataset_name] = data_info
    
    # Combine training datasets from all sources
    from torch.utils.data import ConcatDataset
    combined_train_dataset = ConcatDataset(train_datasets)
    
    # Create combined training DataLoader
    combined_train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n{'='*80}")
    print(f"Multi-Dataset Loading Complete!")
    print(f"  Combined training samples: {len(combined_train_dataset)}")
    print(f"  Combined training batches: {len(combined_train_loader)}")
    print(f"  Individual validation loaders: {len(val_loaders)}")
    print(f"  Individual test loaders: {len(test_loaders)}")
    print(f"{'='*80}\n")
    
    return combined_train_loader, val_loaders, test_loaders, scalers, data_infos


# ==================== Convenience Functions ====================

def get_dataloader(dataset_path, target_H=None, target_W=None, batch_size=32, normalize=True, num_workers=4):
    """
    Convenience function: Load a single dataset and return DataLoaders
    
    This is the main function you'll use for single-dataset training.
    
    Args:
        dataset_path: Path to JSON file (e.g., 'Dataset/BikeNYC_short.json')
        target_H: Target height (None = use original, otherwise resize to this)
        target_W: Target width (None = use original, otherwise resize to this)
        batch_size: Batch size for training
        normalize: Whether to normalize data to [-1, 1]
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader, scaler, data_info
    
    Example:
        >>> train_loader, val_loader, test_loader, scaler, info = get_dataloader(
        ...     'Dataset/BikeNYC_short.json', target_H=32, target_W=32, batch_size=32
        ... )
        >>> for batch in train_loader:
        ...     spatial_temporal = batch['spatial_temporal']  # (B, T, H, W)
        ...     timestamps = batch['timestamps']              # (B, T, 2)
        ...     # forward pass...
    """
    return load_single_dataset(dataset_path, target_H, target_W, batch_size, normalize, num_workers)


def get_multi_dataloader(dataset_paths, target_H, target_W, batch_size=32, normalize=True, num_workers=4):
    """
    Convenience function: Load multiple datasets for multi-dataset training
    
    All datasets are resized to unified dimensions (target_H, target_W).
    
    Args:
        dataset_paths: List of paths to JSON files
        target_H: Unified target height (all datasets resized to this)
        target_W: Unified target width (all datasets resized to this)
        batch_size: Batch size for training
        normalize: Whether to normalize data
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loaders, test_loaders, scalers, data_infos
    
    Example:
        >>> paths = ['Dataset/BikeNYC_short.json', 'Dataset/TaxiNYC_short.json']
        >>> train_loader, val_loaders, test_loaders, scalers, infos = get_multi_dataloader(
        ...     paths, target_H=32, target_W=32
        ... )
        >>> # Train on combined data
        >>> for batch in train_loader:
        ...     # forward pass...
        >>> # Evaluate on each dataset separately
        >>> for val_loader, dataset_name in zip(val_loaders, scalers.keys()):
        ...     # validation for this specific dataset...
    """
    return load_multiple_datasets(dataset_paths, target_H, target_W, batch_size, normalize, num_workers)


def get_few_shot_dataloader(dataset_paths, target_dataset_path, target_ratio, target_H, target_W, 
                            batch_size=32, normalize=True, num_workers=4):
    """
    Create dataloaders for few-shot/zero-shot training
    
    For few-shot: Use 100% of other datasets + target_ratio% of target dataset for training
    For zero-shot (target_ratio=0): Use 100% of other datasets only for training
    
    Args:
        dataset_paths: List of all dataset paths
        target_dataset_path: Path to the target dataset for few-shot evaluation
        target_ratio: Ratio of target training data to use (0.0 for zero-shot, 0.05 for 5%, etc.)
        target_H: Unified target height
        target_W: Unified target width
        batch_size: Batch size for training
        normalize: Whether to normalize data
        num_workers: Number of data loading workers
    
    Returns:
        train_loader: Combined training DataLoader
        target_val_loader: Validation DataLoader for target dataset
        target_test_loader: Test DataLoader for target dataset
        target_scaler: Scaler for target dataset
        target_info: Data info for target dataset
    """
    from torch.utils.data import ConcatDataset, Subset
    
    print(f"\n{'='*80}")
    print(f"Loading Few-Shot Data (target_ratio={target_ratio*100:.1f}%)")
    print(f"Target dataset: {target_dataset_path}")
    print(f"{'='*80}")
    
    train_datasets = []
    target_scaler = None
    target_info = None
    target_val_loader = None
    target_test_loader = None
    
    # Load each dataset
    for json_path in dataset_paths:
        train_loader, val_loader, test_loader, scaler, data_info = load_single_dataset(
            json_path, target_H=target_H, target_W=target_W,
            batch_size=batch_size, normalize=normalize, num_workers=num_workers
        )
        
        if json_path == target_dataset_path:
            # Target dataset
            target_scaler = scaler
            target_info = data_info
            target_val_loader = val_loader
            target_test_loader = test_loader
            
            if target_ratio > 0:
                # Few-shot: use subset of target training data
                full_dataset = train_loader.dataset
                n_samples = len(full_dataset)
                n_subset = max(1, int(n_samples * target_ratio))
                
                # Random subset (fixed seed for reproducibility)
                random.seed(42)
                indices = random.sample(range(n_samples), n_subset)
                subset_dataset = Subset(full_dataset, indices)
                train_datasets.append(subset_dataset)
                print(f"  Target: Using {n_subset}/{n_samples} samples ({target_ratio*100:.1f}%)")
            else:
                # Zero-shot: don't use target training data
                print(f"  Target: Zero-shot mode, excluded from training")
        else:
            # Non-target dataset: use 100% training data
            train_datasets.append(train_loader.dataset)
            print(f"  {data_info['dataset_name']}: Using 100% training data")
    
    # Combine training datasets
    if len(train_datasets) > 0:
        combined_train_dataset = ConcatDataset(train_datasets)
        combined_train_loader = DataLoader(
            combined_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        combined_train_loader = None
    
    print(f"\n{'='*80}")
    print(f"Few-Shot Data Loading Complete!")
    if combined_train_loader:
        print(f"  Combined training samples: {len(combined_train_dataset)}")
        print(f"  Combined training batches: {len(combined_train_loader)}")
    print(f"  Target val batches: {len(target_val_loader) if target_val_loader else 0}")
    print(f"  Target test batches: {len(target_test_loader) if target_test_loader else 0}")
    print(f"{'='*80}\n")
    
    return combined_train_loader, target_val_loader, target_test_loader, target_scaler, target_info

