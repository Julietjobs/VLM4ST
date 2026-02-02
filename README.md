# 🌐 VLM4ST: A Vision-Language Foundation Model for Spatiotemporal Prediction

A novel Vision-Language Model (VLM)-based framework for spatiotemporal prediction tasks, combining multi-scale disentangled attention mechanisms with pre-trained vision-language encoders (ALBEF) and parameter-efficient LoRA fine-tuning.

## 📁 Project Structure

```
VLM4ST/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config/
│   ├── config.yaml                    # Main configuration file
│   ├── config_bert.json               # BERT encoder configuration
│   └── config_no_pretrain.yaml        # Configuration without pretrained weights
├── data/
│   └── dataloader.py                  # Data loading, preprocessing, and normalization
├── Dataset/
│   └── Readme.md                      # Dataset instructions (place datasets here)
├── models/
│   ├── __init__.py                    # Model exports
│   ├── dix_attention.py               # DiX-Attention: Temporal, Spatial, Cross-modal attention
│   ├── prompt_forge.py                # PromptForge: Hierarchical pattern memory
│   ├── gate_fusion.py                 # AdaptiveGatedFusion: Multi-branch feature fusion
│   ├── format_transformer.py          # DualPathFormatTransformer: Image & Sequence conversion
│   ├── vlm_module.py                  # VLMModule: ALBEF-based encoder with LoRA
│   ├── vit.py                         # Vision Transformer implementation
│   ├── xbert.py                       # BERT implementation for ALBEF
│   ├── prediction_head.py             # PredictionHead: Horizon query tokens
│   └── vlm4st.py                      # Main VLM4ST model
├── utils/
│   ├── __init__.py
│   └── normalization.py               # Data normalization utilities
├── train.py                           # Standard training script (100% data)
├── test.py                            # Testing and evaluation script
└── run_few_shot.py                    # Few-shot/Zero-shot training script
```

## 🏗️ Model Architecture

VLM4ST consists of six main modules:

```
Input (B, T_in, H, W)
        │
        ▼
┌───────────────────┐
│  1. DiX-Attention │  → Temporal, Spatial, Cross-modal features
│                   │     + CLS tokens for each modality
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  2. PromptForge   │  → Pattern-aware prompts from hierarchical memory
│                   │     (coarse + fine-grained prototypes)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  3. Gate Fusion   │  → Adaptive weighted fusion of all features
│                   │     (learned gating weights)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  4. Format        │  → Dual-path conversion:
│    Transformer    │     - Image path: (B, 3, 224, 224)
│                   │     - Sequence path: (B, seq_len, 768)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  5. VLM Module    │  → ALBEF visual-language encoding
│    (ALBEF+LoRA)   │     with cross-attention fusion
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  6. Prediction    │  → Horizon query tokens + cross-attention
│       Head        │     + DiX residual connection
└───────────────────┘
        │
        ▼
Output (B, T_out, H, W, C)
```

## 📋 Requirements

### Python Dependencies

```
python>=3.10
transformers>=4.8.1
timm
torch>=2.0.0
torchvision>=0.15.0
numpy
pyyaml
tensorboard
tqdm
einops
scikit-learn
matplotlib
```

### Installation

```bash
# Clone or navigate to the project directory
cd VLM4ST

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch torchvision transformers timm numpy pyyaml tensorboard tqdm einops scikit-learn matplotlib
```

## 📊 Data Format

### Input Requirements

The model expects spatiotemporal data in the UniST format:

- **Input Shape**: `(N, T, H, W)` 
  - `N`: Number of samples
  - `T`: Temporal length (default: 12, using first 6 as input, last 6 as target)
  - `H`: Spatial height (unified to 32 by default)
  - `W`: Spatial width (unified to 32 by default)
- **Normalization**: Data is normalized to `[-1, 1]` using MinMax normalization

## ⚙️ Configuration

Edit `config/config.yaml` to customize model and training settings.

## 💾 Pre-trained Weights

VLM4ST leverages pre-trained ALBEF weights for the VLM module. Download the checkpoint:

```bash
cd checkpoints/pretrained

# Download ALBEF pre-trained checkpoint (14M)
wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth -O checkpoints/pretrained/ALBEF_14M.pth

# Or use the 4M version
wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth -O checkpoints/pretrained/ALBEF_4M.pth
```

Update the path in `config/config.yaml`:

```yaml
model:
  vlm:
    pretrained_path: "./checkpoints/pretrained/ALBEF_14M.pth"
```

The model will automatically load ALBEF weights during initialization.

## 🚀 Training

### Standard Training (Multi-dataset)

```bash
# Multi-dataset training (default)
python train.py --config config/config.yaml --gpu 0

# Single dataset training
python train.py --config config/config.yaml --single_dataset --gpu 0
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to configuration file | `config/config.yaml` |
| `--resume` | Path to checkpoint to resume from | None |
| `--gpu` | GPU device ID | 0 |
| `--single_dataset` | Use single dataset mode | False (multi-dataset) |

### Training Output

Training produces:
- Checkpoints in `./checkpoints/`
- TensorBoard logs in `./logs/all_shot/`
- Training log file with metrics per epoch

## 🧪 Testing

### Basic Testing

```bash
# Multi-dataset testing
python test.py \
  --config config/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --gpu 0

# Single dataset testing
python test.py \
  --config config/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --single_dataset \
  --gpu 0
```

### Testing with Prediction Saving

```bash
python test.py \
  --config config/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --save_predictions \
  --output_dir results \
  --gpu 0
```

### Testing Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to configuration file | `config/config.yaml` |
| `--checkpoint` | Path to model checkpoint | Required |
| `--output_dir` | Directory to save results | `results` |
| `--gpu` | GPU device ID | 0 |
| `--save_predictions` | Save predictions to file | False |
| `--single_dataset` | Use single dataset mode | False |

## 🎯 Few-Shot / Zero-Shot Learning

```bash
# Zero-shot testing (train on other datasets, test on target)
python run_few_shot.py --config config/config.yaml --target_ratio 0.0

# 5% Few-shot (use 5% of target dataset for training)
python run_few_shot.py --config config/config.yaml --target_ratio 0.05

# 10% Few-shot
python run_few_shot.py --config config/config.yaml --target_ratio 0.1

# Specific target dataset
python run_few_shot.py --config config/config.yaml --target_ratio 0.05 \
    --target_dataset "./Dataset/dataset.json"
```

### Few-Shot Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to configuration file | `config/config.yaml` |
| `--target_ratio` | Target dataset ratio | 0.0 (from config) |
| `--gpu` | GPU device ID | 0 |
| `--target_dataset` | Specific target dataset path | All datasets |

## 🔧 Advanced Usage

### Ablation Modes

VLM4ST supports ablation studies with the following modes:

```python
from models.vlm4st import VLM4ST

# Full model (default)
model = VLM4ST(config)

# Without DiX-Attention (only input projection)
model = VLM4ST(config, ablation_mode='no_dix')

# Without PromptForge (zero prompts)
model = VLM4ST(config, ablation_mode='no_prompt')

# Without Sequence path (visual only)
model = VLM4ST(config, ablation_mode='no_seq')

# Without Visual path (sequence only)
model = VLM4ST(config, ablation_mode='no_visual')
```

### LoRA Variants

Configure different LoRA types in `config.yaml`:

```yaml
model:
  vlm:
    use_lora: true
    lora_type: "standard"      # Basic LoRA
    # lora_type: "mtl"         # Multi-Task LoRA with task-specific parameters
    # lora_type: "moe"         # Mixture of Experts LoRA
    # lora_type: "task_adaptive"  # Task-Adaptive LoRA (our proposed)
    lora_rank: 16
    lora_alpha: 32
    num_tasks: 14              # For mtl/task_adaptive
    num_experts: 4             # For moe/task_adaptive
```

### Freeze/Unfreeze Backbone

```python
from models.vlm4st import VLM4ST

model = VLM4ST(config)

# Freeze VLM backbone (LoRA fine-tuning only)
model.freeze_vlm_backbone()

# Unfreeze all parameters (full fine-tuning)
model.unfreeze_all()

# Check parameter counts
param_info = model.get_trainable_parameters()
print(f"Total: {param_info['total']:,}")
print(f"Trainable: {param_info['trainable']:,}")
print(f"Frozen: {param_info['frozen']:,}")
```

### Access Intermediate Outputs

```python
prediction, intermediates = model(X, timestamps=timestamps, return_intermediate=True)

# Available intermediate outputs:
# - X_temporal, X_spatial, X_cross, X_input: DiX-Attention features
# - B_prompt: Generated prompts from PromptForge
# - pattern_info: Pattern weights (coarse/fine) for visualization
# - X_fused: Fused features after gate fusion
# - gate_weights: (B, T, H, W, 4) gating weights
# - image, sequence: Format-transformed VLM inputs
# - vlm_output: VLM encodings (fused_embeds, visual_embeds, text_embeds)
```

### Custom Loss Configuration

```python
loss_config = {
    'mse_weight': 1.0,      # Prediction loss (MSE)
    'mae_weight': 0.5,      # Prediction loss (MAE)
}

loss, loss_dict = model.get_loss(prediction, target, loss_config)
# loss_dict contains: total_loss, mse_loss, mae_loss
```

## 📈 TensorBoard Visualization

View training metrics and visualizations:

```bash
tensorboard --logdir ./logs
```

Available visualizations:
- Training/Validation loss, MSE, MAE (normalized and real-scale)
- Gate weights (temporal, spatial, cross, input)
- Pattern weights (coarse and fine prototypes) heatmaps
- Learning rate schedule

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
TBC……
```

## 📄 License

This project is for research purposes only.

## 🙏 Acknowledgments

- ALBEF model architecture from [ALBEF](https://github.com/salesforce/ALBEF)
- LoRA implementation based on [LoRA paper](https://arxiv.org/abs/2106.09685)
- Dataset format follows [UniST](https://github.com/tsinghua-fib-lab/UniST)
- Vision Transformer from [timm](https://github.com/huggingface/pytorch-image-models)
