"""
Testing script for VLM4ST
"""

import os
import sys
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import time
import psutil
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.tensorboard import SummaryWriter

from models.vlm4st import VLM4ST
from data.dataloader import get_dataloader, get_multi_dataloader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test VLM4ST model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--single_dataset', dest='multi_dataset', action='store_false',
                        help='Use single dataset testing mode (default: multi-dataset)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_model(model, dataloader, device, scaler=None):
    """
    Test the model on the test set
    
    Args:
        model: VLM4ST model
        dataloader: Test dataloader
        device: Device to use
        scaler: Data scaler for inverse transformation
        
    Returns:
        metrics: Dictionary of metrics (Real MAE and Real RMSE)
        all_predictions: All predictions concatenated
        all_targets: All targets concatenated
        avg_gate_weights: Average gate weights (4,) for Temporal, Spatial, Cross-modal, Input
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_gate_weights = torch.zeros(4).to(device)
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            X = batch['spatial_temporal'].to(device)
            timestamps = batch['timestamps'].to(device)
            
            # Split data: input first T_in steps, predict next T_out steps
            # Same as in train.py
            T_in = model.T
            T_out = model.T_out
            T_total = T_in + T_out
            
            # Skip if not enough time steps
            if X.shape[1] < T_total:
                continue
            
            X_input = X[:, :T_in, :, :]  # (B, T_in, H, W) - first T_in steps
            X_target = X[:, T_in:T_total, :, :]  # (B, T_out, H, W) - next T_out steps
            timestamps_input = timestamps[:, :T_in, :]  # (B, T_in, 2)
            
            # Predict with intermediate outputs
            prediction, intermediates = model(X_input, timestamps=timestamps_input, return_intermediate=True)
            gate_weights = intermediates['gate_weights']  # (B, T, H, W, 4)
            
            # Squeeze channel dimension if needed
            if prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            
            # Store predictions and targets
            all_predictions.append(prediction.cpu())
            all_targets.append(X_target.cpu())
            
            # Accumulate gate weights
            total_gate_weights += gate_weights.mean(dim=[0, 1, 2, 3])
            num_batches += 1
    
    # Concatenate all batches
    if len(all_predictions) == 0:
        return {'Real_MAE': 0.0, 'Real_RMSE': 0.0}, None, None, torch.zeros(4)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_gate_weights = total_gate_weights / num_batches
    
    # Compute Real MAE and Real RMSE (in original scale)
    if scaler is not None:
        pred_np = all_predictions.numpy()
        target_np = all_targets.numpy()
        
        # Inverse transform to original scale
        pred_denorm = scaler.inverse_transform(pred_np.reshape(-1, 1)).reshape(pred_np.shape)
        target_denorm = scaler.inverse_transform(target_np.reshape(-1, 1)).reshape(target_np.shape)
        
        # Calculate real-scale metrics
        real_mae = mean_absolute_error(target_denorm.flatten(), pred_denorm.flatten())
        real_mse = mean_squared_error(target_denorm.flatten(), pred_denorm.flatten())
        real_rmse = np.sqrt(real_mse)
    else:
        # If no scaler, compute metrics on normalized data
        real_mae = np.mean(np.abs(all_predictions.numpy() - all_targets.numpy()))
        real_mse = np.mean((all_predictions.numpy() - all_targets.numpy()) ** 2)
        real_rmse = np.sqrt(real_mse)
    
    metrics = {
        'Real_MAE': float(real_mae),
        'Real_RMSE': float(real_rmse)
    }
    
    return metrics, all_predictions, all_targets, avg_gate_weights


def save_results(metrics, predictions, targets, output_dir, dataset_name):
    """Save test results"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, f'{dataset_name}_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Save predictions and targets as numpy arrays
    pred_path = os.path.join(output_dir, f'{dataset_name}_predictions_{timestamp}.npy')
    target_path = os.path.join(output_dir, f'{dataset_name}_targets_{timestamp}.npy')
    
    np.save(pred_path, predictions.numpy())
    np.save(target_path, targets.numpy())
    print(f"Predictions saved to {pred_path}")
    print(f"Targets saved to {target_path}")


def main():
    """Main testing function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get target dimensions from config
    target_H = config['model']['H']
    target_W = config['model']['W']
    
    # Create dataloaders
    print("Creating dataloaders...")
    if args.multi_dataset and 'dataset_paths' in config['data']:
        # Multi-dataset testing mode
        print("Multi-dataset testing mode enabled")
        dataset_paths = config['data']['dataset_paths']
        train_loader, val_loaders, test_loaders, scalers, data_infos = get_multi_dataloader(
            dataset_paths=dataset_paths,
            target_H=target_H,
            target_W=target_W,
            batch_size=config['training'].get('batch_size', 32),
            normalize=True,
            num_workers=config['training'].get('num_workers', 4)
        )
        # Print dataset info
        print(f"\nLoaded {len(data_infos)} datasets for testing:")
        for name, info in data_infos.items():
            print(f"  {name}: {info['test_batches']} test batches")
        
        multi_dataset = True
    else:
        # Single dataset testing mode
        print("Single dataset testing mode")
        dataset_path = config['data']['dataset_path']
        train_loader, val_loader, test_loader, scaler, data_info = get_dataloader(
            dataset_path=dataset_path,
            target_H=target_H,
            target_W=target_W,
            batch_size=config['training'].get('batch_size', 32),
            normalize=True,
            num_workers=config['training'].get('num_workers', 4)
        )
        print(f"\nDataset: {data_info['dataset_name']}")
        print(f"  Test batches: {data_info['test_batches']}")
        
        # Wrap in lists for consistency
        test_loaders = [test_loader]
        scalers = {data_info['dataset_name']: scaler}
        data_infos = {data_info['dataset_name']: data_info}
        multi_dataset = False
    
    # Create model
    print("\nCreating model...")
    model = VLM4ST(config).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    # Create TensorBoard writer for test logs
    log_dir = os.path.join('logs', 'test')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"test_{timestamp}"
    tb_log_path = os.path.join(log_dir, run_name)
    writer = SummaryWriter(tb_log_path)
    
    # Create performance log file in the same directory as TensorBoard logs
    perf_log_path = os.path.join(tb_log_path, 'performance_log.txt')
    perf_log_entries = []
    
    # Test the model on each dataset
    print("\n" + "="*80)
    print("Testing model on datasets:")
    print("="*80)
    
    all_results = {}
    
    for i, (dataset_name, test_loader) in enumerate(zip(data_infos.keys(), test_loaders)):
        print(f"\nTesting on {dataset_name}...")
        dataset_scaler = scalers[dataset_name] if isinstance(scalers, dict) else scalers
        
        # Record memory before inference
        cpu_mem_before = psutil.Process().memory_info().rss / (1024 ** 3)  # GB
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            gpu_mem_before = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        
        # Test with timing
        start_time = time.time()
        metrics, predictions, targets, gate_weights = test_model(model, test_loader, device, dataset_scaler)
        inference_time = time.time() - start_time
        
        # Record memory after inference
        cpu_mem_after = psutil.Process().memory_info().rss / (1024 ** 3)  # GB
        if torch.cuda.is_available():
            gpu_mem_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
            gpu_mem_after = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        else:
            gpu_mem_before = gpu_mem_after = gpu_mem_peak = 0.0
        
        # Log performance entry
        perf_entry = {
            'dataset': dataset_name,
            'inference_time_sec': inference_time,
            'cpu_mem_before_GB': cpu_mem_before,
            'cpu_mem_after_GB': cpu_mem_after,
            'gpu_mem_before_GB': gpu_mem_before,
            'gpu_mem_after_GB': gpu_mem_after,
            'gpu_mem_peak_GB': gpu_mem_peak
        }
        perf_log_entries.append(perf_entry)
        
        # Store results (including gate weights)
        all_results[dataset_name] = {
            **metrics,
            'gate_weights': {
                'temporal': float(gate_weights[0].item()),
                'spatial': float(gate_weights[1].item()),
                'cross': float(gate_weights[2].item()),
                'input': float(gate_weights[3].item())
            }
        }
        
        # Log gate weights to TensorBoard for each dataset
        writer.add_scalar(f'Test/{dataset_name}/gate_weight_temporal', gate_weights[0].item(), 0)
        writer.add_scalar(f'Test/{dataset_name}/gate_weight_spatial', gate_weights[1].item(), 0)
        writer.add_scalar(f'Test/{dataset_name}/gate_weight_cross', gate_weights[2].item(), 0)
        writer.add_scalar(f'Test/{dataset_name}/gate_weight_input', gate_weights[3].item(), 0)
        
        # Print results for this dataset
        print(f"\nResults for {dataset_name}:")
        print("-" * 50)
        print(f"Real MAE:  {metrics['Real_MAE']:.4f}")
        print(f"Real RMSE: {metrics['Real_RMSE']:.4f}")
        print(f"Gate Weights - Temporal: {gate_weights[0]:.4f}, Spatial: {gate_weights[1]:.4f}, Cross: {gate_weights[2]:.4f}, Input: {gate_weights[3]:.4f}")
        print("-" * 50)
        
        # Save predictions if requested
        if args.save_predictions and predictions is not None:
            save_results(metrics, predictions, targets, args.output_dir, dataset_name)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Test Results for All Datasets:")
    print("="*80)
    for dataset_name, result in all_results.items():
        print(f"{dataset_name}:")
        print(f"  Real MAE:  {result['Real_MAE']:.4f}")
        print(f"  Real RMSE: {result['Real_RMSE']:.4f}")
        gw = result['gate_weights']
        print(f"  Gate Weights - Temporal: {gw['temporal']:.4f}, Spatial: {gw['spatial']:.4f}, Cross: {gw['cross']:.4f}, Input: {gw['input']:.4f}")
    
    # Calculate average metrics across all datasets
    if len(all_results) > 1:
        avg_mae = np.mean([m['Real_MAE'] for m in all_results.values()])
        avg_rmse = np.mean([m['Real_RMSE'] for m in all_results.values()])
        avg_gw = {
            'temporal': np.mean([m['gate_weights']['temporal'] for m in all_results.values()]),
            'spatial': np.mean([m['gate_weights']['spatial'] for m in all_results.values()]),
            'cross': np.mean([m['gate_weights']['cross'] for m in all_results.values()]),
            'input': np.mean([m['gate_weights']['input'] for m in all_results.values()])
        }
        print(f"\nAverage across all datasets:")
        print(f"  Real MAE:  {avg_mae:.4f}")
        print(f"  Real RMSE: {avg_rmse:.4f}")
        print(f"  Gate Weights - Temporal: {avg_gw['temporal']:.4f}, Spatial: {avg_gw['spatial']:.4f}, Cross: {avg_gw['cross']:.4f}, Input: {avg_gw['input']:.4f}")
        all_results['Average'] = {'Real_MAE': float(avg_mae), 'Real_RMSE': float(avg_rmse), 'gate_weights': avg_gw}
    
    print("="*80)
    
    # Save all results to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if multi_dataset:
        results_filename = f'multi_dataset_test_results_{timestamp}.json'
    else:
        results_filename = f'{list(data_infos.keys())[0]}_test_results_{timestamp}.json'
    
    results_path = os.path.join(args.output_dir, results_filename)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll results saved to {results_path}")
    
    # Write performance log file
    with open(perf_log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Performance Log - {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        for entry in perf_log_entries:
            f.write(f"Dataset: {entry['dataset']}\n")
            f.write(f"  Inference Time: {entry['inference_time_sec']:.2f} seconds\n")
            f.write(f"  CPU Memory: {entry['cpu_mem_before_GB']:.3f} GB -> {entry['cpu_mem_after_GB']:.3f} GB\n")
            f.write(f"  GPU Memory: {entry['gpu_mem_before_GB']:.3f} GB -> {entry['gpu_mem_after_GB']:.3f} GB (Peak: {entry['gpu_mem_peak_GB']:.3f} GB)\n")
            f.write("-" * 50 + "\n")
        
        # Summary
        total_time = sum(e['inference_time_sec'] for e in perf_log_entries)
        f.write(f"\nTotal Inference Time: {total_time:.2f} seconds\n")
    
    print(f"Performance log saved to {perf_log_path}")
    
    # Close TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to {tb_log_path}")


if __name__ == '__main__':
    main()

