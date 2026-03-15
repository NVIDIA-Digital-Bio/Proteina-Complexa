# Training Guide

> **Note:** This guide is under active development and will be expanded as training workflows are finalized.

This guide covers training Proteina-Complexa models for protein binder design.

> **Documentation Map**
> - Running a design? See [Inference Guide](INFERENCE.md)
> - Tuning YAML configs? See [Configuration Guide](CONFIGURATION_GUIDE.md)
> - Understanding metrics? See [Evaluation Guide](EVALUATION_METRICS.md)
> - Parameter sweeps? See [Sweep System](SWEEP.md)

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Training Configurations](#training-configurations)
4. [Dataset Preparation](#dataset-preparation)
5. [Hyperparameters](#hyperparameters)
6. [Checkpoints](#checkpoints)
7. [Multi-Node Training](#multi-node-training)

---

## Overview

Proteina-Complexa training involves:

1. **Autoencoder Pre-training**: Train the variational autoencoder on protein structures
2. **Flow Matching Training**: Train the generative model with optional target conditioning

### Model Architecture

| Component | Parameters | Description |
|-----------|------------|-------------|
| Flow Model | 160M | Main generative backbone |
| Autoencoder | ~160M | Side-chain latent encoder/decoder |
| Conditioning | Optional | CATH fold conditioning |

---

## Quick Start

### Basic Training

```bash
# Activate environment
source .venv/bin/activate

# Single GPU training (development)
python -m proteinfoundation.train \
    --config-name finetune_local_latents_binder_comb_extra_lenient_pdb \
    +single=true

# Disable logging (faster iteration)
python -m proteinfoundation.train \
    --config-name finetune_local_latents_binder_comb_extra_lenient_pdb \
    +single=true \
    +nolog=true
```

### Multi-GPU Training

```bash
# Multi-node training (requires SLURM)
python -m proteinfoundation.train \
    --config-name finetune_local_latents_binder_comb_extra_lenient_pdb
```

---

## Training Configurations

### Configuration 1: Combination Dataset (Recommended)

**Config**: `finetune_local_latents_binder_comb_extra_lenient_pdb.yaml` -- Fine-tunes the latent-space binder model on a combined AFDB + PDB dataset with relaxed quality filtering.

- **Dataset**: AFDB + CATH dimer + PDB combination
- **Model**: 160M parameter binder model
- **Fold conditioning**: Disabled
- **Hardware**: 12 nodes × 8 GPUs = 96 GPUs

```bash
python -m proteinfoundation.train \
    --config-name finetune_local_latents_binder_comb_extra_lenient_pdb
```

### Configuration 2: CATH-Conditioned

**Config**: `finetune_local_latents_binder_ted_extra_lenient_cat.yaml` -- Fine-tunes on AFDB with TED (domain) filtering and CATH fold-level conditioning for structure-aware generation.

- **Dataset**: AFDB with CATH dimer filtering
- **Model**: 160M with CATH conditioning
- **Fold conditioning**: Enabled
- **Hardware**: 12 nodes × 8 GPUs = 96 GPUs

```bash
python -m proteinfoundation.train \
    --config-name finetune_local_latents_binder_ted_extra_lenient_cat
```

### Other Configurations

| Config | Purpose |
|--------|---------|
| `training_ae_pdb.yaml` | Fine-tune the variational autoencoder on PDB structures (side-chain latent space) |
| `pdb_multimer_chain.yaml` | Extract individual chains from PDB multimers for single-chain training |
| `pdb_multimer_binder_filter.yaml` | Ablation study using filtered PDB binder-like multimers |

---

## Dataset Preparation

### Required Data Structure

```
$DATA_PATH/
├── afdb_preprocessed/
│   └── afdb_cathdimer_extra_lenient/
├── pdb_preprocessed/
│   └── pdb_multimer_filtered/
└── target_data/
    └── [target_sources]/
```

### Dataset Configuration

```yaml
dataset:
  _target_: "proteinfoundation.datasets.AfdbDataset"
  data_dir: ${oc.env:DATA_PATH}/afdb_preprocessed/afdb_cathdimer_extra_lenient
  split: "train"
  
dataloader:
  batch_size: 32
  num_workers: 8
  pin_memory: true
```

---

## Hyperparameters

### Key Training Parameters

```yaml
trainer:
  max_epochs: 500
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  precision: "bf16-mixed"

optimizer:
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  warmup_steps: 1000
  min_lr: 1e-6
```

### Flow Matching Parameters

```yaml
flow_matching:
  sigma_min: 0.001
  sigma_max: 80.0
  schedule: "log"
  
generation:
  args:
    nsteps: 400
    self_cond: true
```

### Model Architecture

```yaml
model:
  hidden_dim: 384
  num_layers: 24
  num_heads: 16
  dropout: 0.0
```

---

## Checkpoints

### Checkpoint Loading

```yaml
# Resume from checkpoint
ckpt_path: /path/to/checkpoints
ckpt_name: checkpoint.ckpt

# Load pretrained autoencoder
autoencoder_ckpt_path: /path/to/ae_checkpoint.ckpt
```

### Checkpoint Saving

```yaml
checkpoint:
  save_top_k: 3
  monitor: "val/loss"
  mode: "min"
  save_last: true
  every_n_train_steps: 10000
```

### Available Pretrained Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `complexa.ckpt` | Main binder model (combination dataset) |
| `complexa_cat.ckpt` | CATH-conditioned model |
| `complexa_ae.ckpt` | Autoencoder (required for all models) |

---

## Multi-Node Training

### SLURM Configuration

```bash
#!/bin/bash
#SBATCH --nodes=12
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:00:00

srun python -m proteinfoundation.train \
    --config-name finetune_local_latents_binder_comb_extra_lenient_pdb
```

### PyTorch Lightning DDP

```yaml
trainer:
  strategy: "ddp"
  devices: 8
  num_nodes: 12
  sync_batchnorm: true
```

### Gradient Accumulation

For memory-constrained setups:
```yaml
trainer:
  accumulate_grad_batches: 4  # Effective batch = batch_size × 4 × num_gpus
```

---

## Monitoring

### Weights & Biases

```yaml
logger:
  _target_: "pytorch_lightning.loggers.WandbLogger"
  project: "proteina-complexa"
  name: ${run_name}
  save_dir: ./logs
```

### TensorBoard

```yaml
logger:
  _target_: "pytorch_lightning.loggers.TensorBoardLogger"
  save_dir: ./logs
  name: ${run_name}
```

### Disable Logging

```bash
python -m proteinfoundation.train \
    --config-name your_config \
    +nolog=true
```

---

## Troubleshooting

### Out of Memory

```yaml
# Reduce batch size
dataloader:
  batch_size: 16

# Enable gradient checkpointing
model:
  gradient_checkpointing: true

# Use mixed precision
trainer:
  precision: "bf16-mixed"
```

### Slow Data Loading

```yaml
dataloader:
  num_workers: 16
  pin_memory: true
  prefetch_factor: 4
```

### NaN Loss

```yaml
# Reduce learning rate
optimizer:
  lr: 5e-5

# Enable gradient clipping
trainer:
  gradient_clip_val: 0.5
```
