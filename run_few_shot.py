"""
Few-shot/Zero-shot training and testing script for VLM4ST

Usage:
    python run_few_shot.py --config config/config.yaml --target_ratio 0.0  # zero-shot
    python run_few_shot.py --config config/config.yaml --target_ratio 0.05  # 5% few-shot
    python run_few_shot.py --config config/config.yaml --target_ratio 0.1   # 10% few-shot
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.vlm4st import VLM4ST
from data.dataloader import get_few_shot_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Few-shot/Zero-shot training for VLM4ST')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--target_ratio', type=float, default=None,
                        help='Target dataset ratio (overrides config). 0.0=zero-shot, 0.05=5%, 0.1=10%')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--target_dataset', type=str, default='./Dataset/BikeNYC2_short.json',
                        help='Specific target dataset path (if not set, runs all datasets)')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model, config):
    """Create optimizer and scheduler"""
    train_config = config['training']
    few_shot_config = config.get('few_shot', {})
    
    lr = train_config['learning_rate']
    weight_decay = train_config.get('weight_decay', 0.05)
    num_epochs = few_shot_config.get('num_epochs', train_config['num_epochs'])
    warmup_epochs = train_config.get('warmup_epochs', 5)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=lr * 0.01
    )
    
    return optimizer, scheduler, warmup_epochs


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warmup for learning rate"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_epoch(model, dataloader, optimizer, device, loss_config, epoch, writer, global_step):
    """Train for one epoch (simplified logging)"""
    model.train()
    
    total_loss = 0
    total_mse = 0
    total_mae = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        X = batch['spatial_temporal'].to(device)
        timestamps = batch['timestamps'].to(device)
        
        T_in = model.T
        T_out = model.T_out
        T_total = T_in + T_out
        
        if X.shape[1] < T_total:
            continue
        
        X_input = X[:, :T_in, :, :]
        X_target = X[:, T_in:T_total, :, :]
        timestamps_input = timestamps[:, :T_in, :]
        
        optimizer.zero_grad()
        
        # Simple forward (no intermediate outputs needed for few-shot)
        prediction = model(X_input, timestamps=timestamps_input)
        
        # Compute loss
        loss, loss_dict = model.get_loss(prediction, X_target, loss_config)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss_dict['total_loss']
        total_mse += loss_dict['mse_loss']
        total_mae += loss_dict['mae_loss']
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'mse': f"{loss_dict['mse_loss']:.4f}",
            'mae': f"{loss_dict['mae_loss']:.4f}"
        })
        
        # Simplified logging: only log at intervals
        if batch_idx % 50 == 0:
            writer.add_scalar('Train/loss_step', loss_dict['total_loss'], global_step)
        
        global_step += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_mse = total_mse / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_mse, avg_mae, global_step


def validate(model, dataloader, device, loss_config, scaler=None):
    """Validate on target dataset"""
    model.eval()
    
    total_loss = 0
    total_mse = 0
    total_mae = 0
    total_real_mae = 0
    total_real_rmse = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            X = batch['spatial_temporal'].to(device)
            timestamps = batch['timestamps'].to(device)
            
            T_in = model.T
            T_out = model.T_out
            T_total = T_in + T_out
            
            if X.shape[1] < T_total:
                continue
            
            X_input = X[:, :T_in, :, :]
            X_target = X[:, T_in:T_total, :, :]
            timestamps_input = timestamps[:, :T_in, :]
            
            prediction = model(X_input, timestamps=timestamps_input)
            
            loss, loss_dict = model.get_loss(prediction, X_target, loss_config)
            
            total_loss += loss_dict['total_loss']
            total_mse += loss_dict['mse_loss']
            total_mae += loss_dict['mae_loss']
            
            # Real-scale metrics
            if scaler is not None:
                pred_np = prediction.squeeze(-1).cpu().numpy() if prediction.shape[-1] == 1 else prediction.cpu().numpy()
                target_np = X_target.cpu().numpy()
                
                pred_denorm = scaler.inverse_transform(pred_np.reshape(-1, 1)).reshape(pred_np.shape)
                target_denorm = scaler.inverse_transform(target_np.reshape(-1, 1)).reshape(target_np.shape)
                
                real_mae = mean_absolute_error(target_denorm.flatten(), pred_denorm.flatten())
                real_rmse = np.sqrt(mean_squared_error(target_denorm.flatten(), pred_denorm.flatten()))
                
                total_real_mae += real_mae
                total_real_rmse += real_rmse
            
            num_batches += 1
    
    if num_batches == 0:
        return 0, 0, 0, 0, 0
    
    return (total_loss / num_batches, total_mse / num_batches, total_mae / num_batches,
            total_real_mae / num_batches, total_real_rmse / num_batches)


def test_model(model, dataloader, device, scaler=None):
    """Test the model on target test set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            X = batch['spatial_temporal'].to(device)
            timestamps = batch['timestamps'].to(device)
            
            T_in = model.T
            T_out = model.T_out
            T_total = T_in + T_out
            
            if X.shape[1] < T_total:
                continue
            
            X_input = X[:, :T_in, :, :]
            X_target = X[:, T_in:T_total, :, :]
            timestamps_input = timestamps[:, :T_in, :]
            
            prediction = model(X_input, timestamps=timestamps_input)
            
            if prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            
            all_predictions.append(prediction.cpu())
            all_targets.append(X_target.cpu())
    
    if len(all_predictions) == 0:
        return {'Real_MAE': 0.0, 'Real_RMSE': 0.0}
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    if scaler is not None:
        pred_np = all_predictions.numpy()
        target_np = all_targets.numpy()
        
        pred_denorm = scaler.inverse_transform(pred_np.reshape(-1, 1)).reshape(pred_np.shape)
        target_denorm = scaler.inverse_transform(target_np.reshape(-1, 1)).reshape(target_np.shape)
        
        real_mae = mean_absolute_error(target_denorm.flatten(), pred_denorm.flatten())
        real_rmse = np.sqrt(mean_squared_error(target_denorm.flatten(), pred_denorm.flatten()))
    else:
        real_mae = np.mean(np.abs(all_predictions.numpy() - all_targets.numpy()))
        real_rmse = np.sqrt(np.mean((all_predictions.numpy() - all_targets.numpy()) ** 2))
    
    return {'Real_MAE': float(real_mae), 'Real_RMSE': float(real_rmse)}


def save_checkpoint(model, optimizer, scheduler, epoch, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    torch.save(checkpoint, save_path)


def run_few_shot_for_target(config, target_dataset_path, target_ratio, device, args):
    """Run few-shot/zero-shot training and testing for a specific target dataset"""
    
    dataset_paths = config['data']['dataset_paths']
    target_name = os.path.basename(target_dataset_path).replace('_short.json', '').replace('.json', '')
    
    # Determine mode string
    if target_ratio == 0:
        mode_str = 'zero_shot'
    else:
        mode_str = f'few_shot_{int(target_ratio*100)}pct'
    
    print(f"\n{'='*80}")
    print(f"Running {mode_str} for target: {target_name}")
    print(f"{'='*80}")
    
    # Setup directories (append /few_shot to base paths from config)
    few_shot_config = config.get('few_shot', {})
    base_save_dir = config['training'].get('save_dir', './checkpoints')
    base_output_dir = config['testing'].get('output_dir', './results')
    base_log_dir = config['training'].get('log_dir', './logs')
    
    save_dir = os.path.join(base_save_dir, 'few_shot')
    output_dir = os.path.join(base_output_dir, 'few_shot')
    log_dir = os.path.join(base_log_dir, 'few_shot')
    
    target_save_dir = os.path.join(save_dir, mode_str, target_name)
    target_output_dir = os.path.join(output_dir, mode_str)
    target_log_dir = os.path.join(log_dir, f'{mode_str}_{target_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    os.makedirs(target_save_dir, exist_ok=True)
    os.makedirs(target_output_dir, exist_ok=True)
    os.makedirs(target_log_dir, exist_ok=True)
    
    # Create tensorboard writer (simplified)
    writer = SummaryWriter(target_log_dir)
    
    # Get target dimensions
    target_H = config['model']['H']
    target_W = config['model']['W']
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, scaler, data_info = get_few_shot_dataloader(
        dataset_paths=dataset_paths,
        target_dataset_path=target_dataset_path,
        target_ratio=target_ratio,
        target_H=target_H,
        target_W=target_W,
        batch_size=config['training'].get('batch_size', 32),
        normalize=True,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    if train_loader is None:
        print("Error: No training data available!")
        return None
    
    # Create model
    print("\nCreating model...")
    model = VLM4ST(config).to(device)
    
    # Create optimizer
    optimizer, scheduler, warmup_epochs = create_optimizer(model, config)
    
    # Training loop
    num_epochs = few_shot_config.get('num_epochs', config['training']['num_epochs'])
    loss_config = config['training']['loss']
    
    best_val_loss = float('inf')
    best_epoch = 0
    global_step = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Warmup
        if epoch < warmup_epochs:
            warmup_lr(optimizer, epoch, warmup_epochs, config['training']['learning_rate'])
        
        # Train
        train_loss, train_mse, train_mae, global_step = train_epoch(
            model, train_loader, optimizer, device, loss_config, epoch, writer, global_step
        )
        
        # Validate on target dataset
        val_loss, val_mse, val_mae, val_real_mae, val_real_rmse = validate(
            model, val_loader, device, loss_config, scaler
        )
        
        # Update scheduler
        if scheduler and epoch >= warmup_epochs:
            scheduler.step()
        
        # Simplified logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val Real MAE={val_real_mae:.2f}, Val Real RMSE={val_real_rmse:.2f}")
        
        writer.add_scalar('Train/loss_epoch', train_loss, epoch)
        writer.add_scalar('Val/loss_epoch', val_loss, epoch)
        writer.add_scalar('Val/real_mae_epoch', val_real_mae, epoch)
        writer.add_scalar('Val/real_rmse_epoch', val_real_rmse, epoch)
        writer.add_scalar('Train/lr', current_lr, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = os.path.join(target_save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
    
    # Save last model
    last_path = os.path.join(target_save_dir, 'last_model.pth')
    save_checkpoint(model, optimizer, scheduler, num_epochs - 1, last_path)
    print(f"Saved last model to {last_path}")
    
    writer.close()
    
    # Test with best model
    print(f"\nTesting with best model (epoch {best_epoch})...")
    best_checkpoint = torch.load(os.path.join(target_save_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = test_model(model, test_loader, device, scaler)
    
    print(f"\nTest Results for {target_name} ({mode_str}):")
    print(f"  Real MAE:  {test_metrics['Real_MAE']:.4f}")
    print(f"  Real RMSE: {test_metrics['Real_RMSE']:.4f}")
    
    # Save results
    results = {
        'target_dataset': target_name,
        'mode': mode_str,
        'target_ratio': target_ratio,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_metrics': test_metrics
    }
    
    results_path = os.path.join(target_output_dir, f'{target_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    
    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override target_ratio from command line if provided
    target_ratio = args.target_ratio
    if target_ratio is None:
        target_ratio = config.get('few_shot', {}).get('target_ratio', 0.0)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_paths = config['data']['dataset_paths']
    
    # Determine which datasets to run
    if args.target_dataset:
        # Run for specific target dataset
        if args.target_dataset not in dataset_paths:
            print(f"Error: Target dataset {args.target_dataset} not in dataset_paths!")
            return
        target_datasets = [args.target_dataset]
    else:
        # Run for all datasets
        target_datasets = dataset_paths
    
    # Run few-shot/zero-shot for each target
    all_results = {}
    
    for target_path in target_datasets:
        result = run_few_shot_for_target(config, target_path, target_ratio, device, args)
        if result:
            all_results[result['target_dataset']] = result
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - All Results:")
    print("="*80)
    
    mode_str = 'zero_shot' if target_ratio == 0 else f'few_shot_{int(target_ratio*100)}pct'
    
    for name, result in all_results.items():
        metrics = result['test_metrics']
        print(f"{name}:")
        print(f"  Real MAE:  {metrics['Real_MAE']:.4f}")
        print(f"  Real RMSE: {metrics['Real_RMSE']:.4f}")
    
    if len(all_results) > 1:
        avg_mae = np.mean([r['test_metrics']['Real_MAE'] for r in all_results.values()])
        avg_rmse = np.mean([r['test_metrics']['Real_RMSE'] for r in all_results.values()])
        print(f"\nAverage across all datasets:")
        print(f"  Real MAE:  {avg_mae:.4f}")
        print(f"  Real RMSE: {avg_rmse:.4f}")
        all_results['Average'] = {'Real_MAE': float(avg_mae), 'Real_RMSE': float(avg_rmse)}
    
    print("="*80)
    
    # Save all results summary
    base_output_dir = config['testing'].get('output_dir', './results')
    output_dir = os.path.join(base_output_dir, 'few_shot')
    summary_dir = os.path.join(output_dir, mode_str)
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_path = os.path.join(summary_dir, f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4, default=str)
    print(f"\nAll results summary saved to {summary_path}")


if __name__ == '__main__':
    main()

