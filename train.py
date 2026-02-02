"""
Training script for VLM4ST
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from models.vlm4st import VLM4ST
from data.dataloader import get_dataloader, get_multi_dataloader


def log_pattern_weights_heatmap(writer, pattern_info, epoch, dataset_name=''):
    """Log pattern weights as heatmaps to TensorBoard for validation
    
    Args:
        pattern_info: Dict with averaged weights (no batch dimension)
            coarse_w: (num_coarse,), fine_w: (num_coarse, num_fine)
    """
    modalities = ['temporal', 'spatial', 'cross']
    prefix = f'Val/{dataset_name}' if dataset_name else 'Val'
    
    for modality in modalities:
        coarse_w, fine_w = pattern_info[f'{modality}_weights']
        coarse_w = coarse_w.detach().cpu().numpy()  # (num_coarse,)
        fine_w = fine_w.detach().cpu().numpy()      # (num_coarse, num_fine)
        # Combined weights
        combined_w = coarse_w[:, np.newaxis] * fine_w  # (num_coarse, num_fine)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(combined_w, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xlabel('Fine Prototype')
        ax.set_ylabel('Coarse Prototype')
        ax.set_xticks(range(fine_w.shape[1]))
        ax.set_yticks(range(fine_w.shape[0]))
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        
        writer.add_figure(f'{prefix}/{modality}/pattern_weights_heatmap', fig, epoch)
        plt.close(fig)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train VLM4ST model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--single_dataset', dest='multi_dataset', action='store_false',
                        help='Use single dataset training mode (default: multi-dataset)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_training(config, args):
    """Setup training environment"""
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    save_dir = config['training']['save_dir']
    log_dir = os.path.join(config['training']['log_dir'], 'all_shot')  # train.py uses 100% training data
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create tensorboard writer
    # Extract dataset name from config
    if args.multi_dataset and 'dataset_paths' in config['data']:
        dataset_name = 'multi_dataset'
    else:
        dataset_name = os.path.basename(config['data']['dataset_path']).replace('_short.json', '').replace('.json', '')
    run_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_log_dir = os.path.join(log_dir, run_name)
    writer = SummaryWriter(run_log_dir)
    
    # Create log file
    log_file_path = os.path.join(run_log_dir, 'training_log.txt')
    
    return device, writer, run_name, log_file_path


def create_optimizer(model, config):
    """Create optimizer and scheduler"""
    train_config = config['training']
    
    optimizer_type = train_config.get('optimizer', 'adamw').lower()
    lr = train_config['learning_rate']
    weight_decay = train_config.get('weight_decay', 0.05)
    
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Create scheduler
    scheduler_type = train_config.get('scheduler', 'cosine').lower()
    num_epochs = train_config['num_epochs']
    warmup_epochs = train_config.get('warmup_epochs', 5)
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=lr * 0.01
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1
        )
    else:
        scheduler = None
    
    return optimizer, scheduler, warmup_epochs


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warmup for learning rate"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_epoch(model, dataloader, optimizer, device, loss_config, epoch, writer, global_step, scaler=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_mse = 0
    total_mae = 0
    total_real_mae = 0
    total_real_rmse = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        X = batch['spatial_temporal'].to(device)
        timestamps = batch['timestamps'].to(device)
        
        # Forward pass
        # Split data: input first T_in steps, predict next T_out steps
        # X shape: (B, T_total, H, W) where T_total >= T_in + T_out
        T_in = model.T
        T_out = model.T_out
        T_total = T_in + T_out
        
        # Skip if not enough time steps
        if X.shape[1] < T_total:
            continue
        
        X_input = X[:, :T_in, :, :]  # (B, T_in, H, W) - first T_in steps
        X_target = X[:, T_in:T_total, :, :]  # (B, T_out, H, W) - next T_out steps
        timestamps_input = timestamps[:, :T_in, :]  # (B, T_in, 2)
        
        optimizer.zero_grad()
        
        # Model prediction with intermediate outputs
        prediction, intermediates = model(X_input, timestamps=timestamps_input, return_intermediate=True)
        gate_weights = intermediates['gate_weights']  # (B, T, H, W, 4)
        pattern_info = intermediates['pattern_info']
        
        # Compute loss
        loss, loss_dict = model.get_loss(prediction, X_target, loss_config)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics (normalized space)
        total_loss += loss_dict['total_loss']
        total_mse += loss_dict['mse_loss']
        total_mae += loss_dict['mae_loss']
        
        # Compute real-scale metrics (for monitoring only, no gradient)
        if scaler is not None:
            with torch.no_grad():
                # Squeeze channel dimension if needed
                pred_np = prediction.squeeze(-1).detach().cpu().numpy() if prediction.shape[-1] == 1 else prediction.detach().cpu().numpy()
                target_np = X_target.detach().cpu().numpy()
                
                # Inverse transform to original scale
                pred_denorm = scaler.inverse_transform(pred_np.reshape(-1, 1)).reshape(pred_np.shape)
                target_denorm = scaler.inverse_transform(target_np.reshape(-1, 1)).reshape(target_np.shape)
                
                # Calculate real-scale metrics
                real_mae = mean_absolute_error(target_denorm.flatten(), pred_denorm.flatten())
                real_mse = mean_squared_error(target_denorm.flatten(), pred_denorm.flatten())
                real_rmse = np.sqrt(real_mse)
                
                total_real_mae += real_mae
                total_real_rmse += real_rmse
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'mse': f"{loss_dict['mse_loss']:.4f}",
            'mae': f"{loss_dict['mae_loss']:.4f}"
        })
        
        # Log to tensorboard
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/loss_step', loss_dict['total_loss'], global_step)
            writer.add_scalar('Train/mse_step', loss_dict['mse_loss'], global_step)
            writer.add_scalar('Train/mae_step', loss_dict['mae_loss'], global_step)
            if scaler is not None:
                writer.add_scalar('Train/real_mae_step', real_mae, global_step)
                writer.add_scalar('Train/real_rmse_step', real_rmse, global_step)
            # Log gate weights (average across batch, time, height, width)
            avg_gate_weights = gate_weights.mean(dim=[0, 1, 2, 3]).detach().cpu()  # (4,)
            writer.add_scalar('Train/gate_weight_temporal', avg_gate_weights[0].item(), global_step)
            writer.add_scalar('Train/gate_weight_spatial', avg_gate_weights[1].item(), global_step)
            writer.add_scalar('Train/gate_weight_cross', avg_gate_weights[2].item(), global_step)
            writer.add_scalar('Train/gate_weight_input', avg_gate_weights[3].item(), global_step)
            # Log pattern weights histogram
            for modality in ['temporal', 'spatial', 'cross']:
                coarse_w, fine_w = pattern_info[f'{modality}_weights']
                writer.add_histogram(f'Train/{modality}_coarse_weights', coarse_w.detach().cpu(), global_step)
                writer.add_histogram(f'Train/{modality}_fine_weights', fine_w.detach().cpu(), global_step)
        
        global_step += 1
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    avg_real_mae = total_real_mae / num_batches if scaler is not None else 0
    avg_real_rmse = total_real_rmse / num_batches if scaler is not None else 0
    
    return avg_loss, avg_mse, avg_mae, global_step, avg_real_mae, avg_real_rmse


def validate(model, dataloader, device, loss_config, scaler=None):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    total_mse = 0
    total_mae = 0
    total_real_mae = 0
    total_real_rmse = 0
    total_gate_weights = torch.zeros(4).to(device)
    # Accumulators for pattern weights (initialized on first batch)
    total_pattern_weights = None
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            X = batch['spatial_temporal'].to(device)
            timestamps = batch['timestamps'].to(device)
            
            # Split data: input first T_in steps, predict next T_out steps
            T_in = model.T
            T_out = model.T_out
            T_total = T_in + T_out
            
            # Skip if not enough time steps
            if X.shape[1] < T_total:
                continue
            
            X_input = X[:, :T_in, :, :]  # (B, T_in, H, W)
            X_target = X[:, T_in:T_total, :, :]  # (B, T_out, H, W)
            timestamps_input = timestamps[:, :T_in, :]  # (B, T_in, 2)
            
            # Predict with intermediate outputs
            prediction, intermediates = model(X_input, timestamps=timestamps_input, return_intermediate=True)
            gate_weights = intermediates['gate_weights']  # (B, T, H, W, 4)
            pattern_info = intermediates['pattern_info']
            
            # Accumulate pattern weights (average across batch for each batch)
            if total_pattern_weights is None:
                total_pattern_weights = {
                    f'{m}_weights': (
                        pattern_info[f'{m}_weights'][0].mean(dim=0),  # coarse: (num_coarse,)
                        pattern_info[f'{m}_weights'][1].mean(dim=0)   # fine: (num_coarse, num_fine)
                    ) for m in ['temporal', 'spatial', 'cross']
                }
            else:
                for m in ['temporal', 'spatial', 'cross']:
                    coarse_w, fine_w = pattern_info[f'{m}_weights']
                    total_pattern_weights[f'{m}_weights'] = (
                        total_pattern_weights[f'{m}_weights'][0] + coarse_w.mean(dim=0),
                        total_pattern_weights[f'{m}_weights'][1] + fine_w.mean(dim=0)
                    )
            
            # Compute loss (normalized space)
            loss, loss_dict = model.get_loss(prediction, X_target, loss_config)
            
            total_loss += loss_dict['total_loss']
            total_mse += loss_dict['mse_loss']
            total_mae += loss_dict['mae_loss']
            
            # Accumulate gate weights
            total_gate_weights += gate_weights.mean(dim=[0, 1, 2, 3])
            
            # Compute real-scale metrics
            if scaler is not None:
                # Squeeze channel dimension if needed
                pred_np = prediction.squeeze(-1).cpu().numpy() if prediction.shape[-1] == 1 else prediction.cpu().numpy()
                target_np = X_target.cpu().numpy()
                
                # Inverse transform to original scale
                pred_denorm = scaler.inverse_transform(pred_np.reshape(-1, 1)).reshape(pred_np.shape)
                target_denorm = scaler.inverse_transform(target_np.reshape(-1, 1)).reshape(target_np.shape)
                
                # Calculate real-scale metrics
                real_mae = mean_absolute_error(target_denorm.flatten(), pred_denorm.flatten())
                real_mse = mean_squared_error(target_denorm.flatten(), pred_denorm.flatten())
                real_rmse = np.sqrt(real_mse)
                
                total_real_mae += real_mae
                total_real_rmse += real_rmse
            
            num_batches += 1
    
    if num_batches == 0:
        return 0, 0, 0, 0, 0, torch.zeros(4), None
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    avg_real_mae = total_real_mae / num_batches if scaler is not None else 0
    avg_real_rmse = total_real_rmse / num_batches if scaler is not None else 0
    avg_gate_weights = total_gate_weights / num_batches
    
    # Average pattern weights
    avg_pattern_info = {
        f'{m}_weights': (
            total_pattern_weights[f'{m}_weights'][0] / num_batches,
            total_pattern_weights[f'{m}_weights'][1] / num_batches
        ) for m in ['temporal', 'spatial', 'cross']
    } if total_pattern_weights is not None else None
    
    return avg_loss, avg_mse, avg_mae, avg_real_mae, avg_real_rmse, avg_gate_weights, avg_pattern_info


def save_checkpoint(model, optimizer, scheduler, epoch, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup training
    device, writer, run_name, log_file_path = setup_training(config, args)
    
    # Get target dimensions from config
    target_H = config['model']['H']
    target_W = config['model']['W']
    
    # Create dataloaders
    print("Creating dataloaders...")
    if args.multi_dataset and 'dataset_paths' in config['data']:
        # Multi-dataset training mode
        print("Multi-dataset training mode enabled")
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
        print(f"\nLoaded {len(data_infos)} datasets, unified to ({target_H}, {target_W}):")
        for name, info in data_infos.items():
            print(f"  {name}: {info['H_original']}×{info['W_original']} -> {info['H']}×{info['W']}")
        scaler = scalers
        val_loader = val_loaders
        test_loader = test_loaders
    else:
        # Single dataset training mode
        print("Single dataset training mode")
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
        if data_info['H_original'] != data_info['H'] or data_info['W_original'] != data_info['W']:
            print(f"  Resized from ({data_info['H_original']}, {data_info['W_original']}) to ({data_info['H']}, {data_info['W']})")
    
    # Create model (VLM module will automatically load pretrained weights if configured)
    print("\nCreating model...")
    model = VLM4ST(config).to(device)
    
    # Print parameter count
    param_info = model.get_trainable_parameters()
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    print(f"Frozen parameters: {param_info['frozen']:,}")
    
    # Create optimizer and scheduler
    optimizer, scheduler, warmup_epochs = create_optimizer(model, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print("Starting training...")
    num_epochs = config['training']['num_epochs']
    save_frequency = config['training']['save_frequency']
    loss_config = config['training']['loss']
    
    best_val_loss = float('inf')
    global_step = 0
    
    # Initialize log file
    with open(log_file_path, 'w') as log_f:
        log_f.write(f"Training Log - {run_name}\n")
        log_f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write("="*80 + "\n\n")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Warmup
        if epoch < warmup_epochs:
            warmup_lr(optimizer, epoch, warmup_epochs, config['training']['learning_rate'])
        
        # Train
        # For multi-dataset, scaler is a dict, so pass None for training (cannot determine which scaler to use for mixed batches)
        train_scaler = None if isinstance(scaler, dict) else scaler
        train_loss, train_mse, train_mae, global_step, train_real_mae, train_real_rmse = train_epoch(
            model, train_loader, optimizer, device, loss_config,
            epoch, writer, global_step, train_scaler
        )
        
        # Validate
        if isinstance(val_loader, list):
            # Multi-dataset: validate on each dataset separately
            val_losses = []
            val_mses = []
            val_maes = []
            val_real_maes = []
            val_real_rmses = []
            val_gate_weights_list = []
            print("\nValidation on individual datasets:")
            for i, vloader in enumerate(val_loader):
                dataset_name = list(data_infos.keys())[i]
                dataset_scaler = scalers[dataset_name] if isinstance(scalers, dict) else scalers
                v_loss, v_mse, v_mae, v_real_mae, v_real_rmse, v_gate_weights, v_pattern_info = validate(model, vloader, device, loss_config, dataset_scaler)
                val_losses.append(v_loss)
                val_mses.append(v_mse)
                val_maes.append(v_mae)
                val_real_maes.append(v_real_mae)
                val_real_rmses.append(v_real_rmse)
                val_gate_weights_list.append(v_gate_weights)
                print(f"  {dataset_name}: Loss={v_loss:.4f}, MSE={v_mse:.4f}, MAE={v_mae:.4f}, Real MAE={v_real_mae:.2f}, Real RMSE={v_real_rmse:.2f}")
                # Log per-dataset metrics to tensorboard
                writer.add_scalar(f'Val/{dataset_name}/real_mae_epoch', v_real_mae, epoch)
                writer.add_scalar(f'Val/{dataset_name}/real_rmse_epoch', v_real_rmse, epoch)
                # Log pattern weights heatmap for each dataset
                if v_pattern_info is not None:
                    log_pattern_weights_heatmap(writer, v_pattern_info, epoch, dataset_name)
            # Use average validation loss
            val_loss = np.mean(val_losses)
            val_mse = np.mean(val_mses)
            val_mae = np.mean(val_maes)
            val_real_mae = np.mean(val_real_maes)
            val_real_rmse = np.mean(val_real_rmses)
            val_gate_weights = torch.stack(val_gate_weights_list).mean(dim=0)
            print(f"  Average: Loss={val_loss:.4f}, MSE={val_mse:.4f}, MAE={val_mae:.4f}, Real MAE={val_real_mae:.2f}, Real RMSE={val_real_rmse:.2f}")
        else:
            # Single dataset
            val_loss, val_mse, val_mae, val_real_mae, val_real_rmse, val_gate_weights, val_pattern_info = validate(model, val_loader, device, loss_config, scaler)
            # Log pattern weights heatmap
            if val_pattern_info is not None:
                log_pattern_weights_heatmap(writer, val_pattern_info, epoch)
        
        # Update scheduler
        if scheduler and epoch >= warmup_epochs:
            scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
        if train_scaler is not None:
            print(f"Train Real MAE: {train_real_mae:.2f}, Real RMSE: {train_real_rmse:.2f}")
        print(f"Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
        print(f"Val Real MAE: {val_real_mae:.2f}, Real RMSE: {val_real_rmse:.2f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Write to log file
        with open(log_file_path, 'a') as log_f:
            log_f.write(f"Epoch {epoch}/{num_epochs} - LR: {current_lr:.6f}\n")
            log_f.write(f"  Train: Loss={train_loss:.4f}, MSE={train_mse:.4f}, MAE={train_mae:.4f}")
            if train_scaler is not None:
                log_f.write(f", Real MAE={train_real_mae:.2f}, Real RMSE={train_real_rmse:.2f}")
            log_f.write("\n")
            
            # Log validation results
            if isinstance(val_loader, list):
                # Multi-dataset: log each dataset separately
                log_f.write(f"  Val (Avg): Loss={val_loss:.4f}, MSE={val_mse:.4f}, MAE={val_mae:.4f}, Real MAE={val_real_mae:.2f}, Real RMSE={val_real_rmse:.2f}\n")
                for i, dataset_name in enumerate(data_infos.keys()):
                    log_f.write(f"    - {dataset_name}: Real MAE={val_real_maes[i]:.2f}, Real RMSE={val_real_rmses[i]:.2f}\n")
            else:
                # Single dataset
                log_f.write(f"  Val: Loss={val_loss:.4f}, MSE={val_mse:.4f}, MAE={val_mae:.4f}, Real MAE={val_real_mae:.2f}, Real RMSE={val_real_rmse:.2f}\n")
            log_f.write("\n")
        
        writer.add_scalar('Train/loss_epoch', train_loss, epoch)
        writer.add_scalar('Train/mse_epoch', train_mse, epoch)
        writer.add_scalar('Train/mae_epoch', train_mae, epoch)
        writer.add_scalar('Val/loss_epoch', val_loss, epoch)
        writer.add_scalar('Val/mse_epoch', val_mse, epoch)
        writer.add_scalar('Val/mae_epoch', val_mae, epoch)
        writer.add_scalar('Train/lr', current_lr, epoch)
        if train_scaler is not None:
            writer.add_scalar('Train/real_mae_epoch', train_real_mae, epoch)
            writer.add_scalar('Train/real_rmse_epoch', train_real_rmse, epoch)
        writer.add_scalar('Val/real_mae_epoch', val_real_mae, epoch)
        writer.add_scalar('Val/real_rmse_epoch', val_real_rmse, epoch)
        # Log validation gate weights
        writer.add_scalar('Val/gate_weight_temporal', val_gate_weights[0].item(), epoch)
        writer.add_scalar('Val/gate_weight_spatial', val_gate_weights[1].item(), epoch)
        writer.add_scalar('Val/gate_weight_cross', val_gate_weights[2].item(), epoch)
        writer.add_scalar('Val/gate_weight_input', val_gate_weights[3].item(), epoch)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if (epoch + 1) % save_frequency == 0 or is_best:
            save_path = os.path.join(
                config['training']['save_dir'],
                f"{run_name}_epoch{epoch}.pth"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, save_path, is_best)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Write final summary to log file
    with open(log_file_path, 'a') as log_f:
        log_f.write("="*80 + "\n")
        log_f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Best validation loss: {best_val_loss:.4f}\n")
        log_f.write("="*80 + "\n")
    
    writer.close()


if __name__ == '__main__':
    main()

