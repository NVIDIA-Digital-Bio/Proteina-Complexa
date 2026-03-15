# Concat Features System for Protein Foundation Models

## Overview

The concat features system enables efficient conditioning on motifs and targets for protein design tasks. It works by temporarily extending sequence representations with compact auxiliary features (motifs/targets), processing them through the neural network, and then trimming back to the original sequence length.

## Key Benefits

- **Efficient Memory Usage**: Only actual motif/target residues are processed (compact representation)
- **Seamless Integration**: Automatic extension/trimming without manual dimension management
- **Flexible Conditioning**: Support for motif-based and target-based conditioning
- **Cross-Sequence Features**: Advanced pair features for motif/target interactions

## Architecture Overview

```
Original Sequence [b, n, d]
         ↓
    Extend with Compact Features
         ↓
Extended Sequence [b, n+m, d]
         ↓
    Neural Network Processing
         ↓
    Trim to Original Length
         ↓
Output Sequence [b, n, d]
```

## Data Preparation Pipeline

### 1. Transform Pipeline

**Motif Preparation:**
```python
# transforms.py
MotifMaskTransform() → motif_mask [n, 37]
ExtractMotifCoordinatesTransform(compact_mode=True) → x_motif [n_motif, 37, 3], seq_motif [n_motif]
```

**Target Preparation:**
```python  
# transforms.py
TargetMaskTransform() → target_mask [n, 37]
ExtractTargetCoordinatesTransform(compact_mode=True) → x_target [n_target, 37, 3], seq_target [n_target]
FilterTargetResiduesTransform() → filters main features to binder residues only
```

### 2. Batch Structure

After transforms, the batch contains:
```python
batch = {
    # Main sequence features (binder residues only after filtering)
    "coords_nm": [b, n_binder, 37, 3],
    "residue_type": [b, n_binder],
    "mask": [b, n_binder],
    
    # Compact motif features (if enabled)
    "x_motif": [b, n_motif, 37, 3],          # Compact motif coordinates
    "motif_mask": [b, n_motif, 37],          # Compact motif atom mask
    "seq_motif": [b, n_motif],               # Compact motif sequence
    "seq_motif_mask": [b, n_motif],          # Compact motif residue mask
    
    # Compact target features (if enabled)
    "x_target": [b, n_target, 37, 3],        # Compact target coordinates  
    "target_mask": [b, n_target, 37],        # Compact target atom mask
    "seq_target": [b, n_target],             # Compact target sequence
    "seq_target_mask": [b, n_target],        # Compact target residue mask
}
```

## Feature Computation

### 1. Individual Feature Classes

**MotifConcatSeqFeat**: Combines motif coordinates, sequence, and masks
```python
def forward(self, batch):
    # Extract compact motif features
    coords_feats = Atom37NanometersCoorsSeqFeat()(motif_coords)     # [b, n_motif, 148]
    seq_feats = ResidueTypeSeqFeat()(motif_sequence)               # [b, n_motif, 20]
    mask_feats = motif_mask                                        # [b, n_motif, 37]
    
    # Concatenate features  
    combined_feats = torch.cat([coords_feats, seq_feats, mask_feats], dim=-1)  # [b, n_motif, 205]
    return combined_feats, motif_residue_mask
```

**TargetConcatSeqFeat**: Similar structure for target features
```python
# Same pattern but for target data
combined_feats: [b, n_target, 205]  # 148 + 20 + 37 = 205
```

### 2. ConcatFeaturesFactory

Coordinates multiple concat feature types:
```python
class ConcatFeaturesFactory:
    def forward(self, batch, seq_repr, seq_mask):
        # Get compact features from enabled creators
        all_feats = []
        all_masks = []
        
        if enable_motif:
            motif_feats, motif_mask = MotifConcatSeqFeat()(batch)  # [b, n_motif, 205]
            all_feats.append(motif_feats)
            all_masks.append(motif_mask)
            
        if enable_target:
            target_feats, target_mask = TargetConcatSeqFeat()(batch)  # [b, n_target, 205]
            all_feats.append(target_feats)  
            all_masks.append(target_mask)
        
        # Concatenate along sequence dimension
        combined_feats = torch.cat(all_feats, dim=1)  # [b, n_motif + n_target, 205]
        combined_masks = torch.cat(all_masks, dim=1)  # [b, n_motif + n_target]
        
        # Project to match sequence representation dimension
        projected_feats = self.linear_out(combined_feats)  # [b, n_concat, token_dim]
        
        # Extend original sequence representation
        extended_seq_repr = torch.cat([seq_repr, projected_feats], dim=1)  # [b, n + n_concat, token_dim]
        extended_mask = torch.cat([seq_mask, combined_masks], dim=1)      # [b, n + n_concat]
        
        return extended_seq_repr, extended_mask
```

## Neural Network Integration

### LocalLatentsTransformer

```python
def forward(self, input):
    # 1. Initial sequence representation
    seq_repr = self.init_repr_factory(input)  # [b, n, token_dim]
    mask = input["mask"]                      # [b, n]
    b, n_orig, _ = seq_repr.shape
    
    # 2. Extend with concat features (if enabled)
    if self.use_concat:
        seq_repr, mask = self.concat_factory(input, seq_repr, mask)  
        # seq_repr: [b, n + n_concat, token_dim]
        # mask: [b, n + n_concat]
        n_concat = seq_repr.shape[1] - n_orig
        
        # Extend conditioning for concat residues
        if n_concat > 0:
            zero_cond = torch.zeros(b, n_concat, conditioning.shape[-1])
            conditioning = torch.cat([conditioning, zero_cond], dim=1)
    
    # 3. Pair representation handling
    if self.use_advanced_pair and n_concat > 0:
        # Advanced: compute cross-sequence pair features
        pair_repr = self.pair_repr_builder(input)                    # [b, n_orig, n_orig, pair_dim]
        pair_repr = self.concat_pair_factory(input, pair_repr)       # [b, n_extended, n_extended, pair_dim]
    else:
        # Simple: zero-pad pair representation
        pair_repr = self.pair_repr_builder(input)  # [b, n_orig, n_orig, pair_dim]
        if n_concat > 0:
            # Extend with zeros to [b, n_extended, n_extended, pair_dim]
            pair_repr = self._zero_pad_pair_representation(pair_repr, n_concat)
    
    # 4. Process through transformer layers
    for layer in self.transformer_layers:
        seq_repr = layer(seq_repr, pair_repr, conditioning, mask)  # [b, n_extended, token_dim]
    
    # 5. Generate outputs
    local_latents_out = self.local_latents_linear(seq_repr)  # [b, n_extended, latent_dim]
    ca_out = self.ca_linear(seq_repr)                        # [b, n_extended, 3]
    
    # 6. Trim back to original sequence length
    if n_concat > 0:
        local_latents_out = local_latents_out[:, :n_orig, :]  # [b, n_orig, latent_dim]
        ca_out = ca_out[:, :n_orig, :]                        # [b, n_orig, 3]
    
    return {"local_latents": local_latents_out, "bb_ca": ca_out}
```

## Advanced Pair Features

### ConcatPairFeaturesFactory

For advanced conditioning, creates cross-sequence pair features:

```python
# Block matrix structure for extended pair representation:
# 
#                    Original    Concat
#                   [n_orig]   [n_concat]
#     Original  ┌─────────────┬─────────────┐
#     [n_orig]  │   Original  │  Orig→Concat│
#               │   Pair Rep  │  Features   │
#               ├─────────────┼─────────────┤
#     Concat    │ Concat→Orig │ Concat→Concat│
#    [n_concat] │  Features   │  Features   │  
#               └─────────────┴─────────────┘

class ConcatPairFeaturesFactory:
    def forward(self, batch, orig_pair_rep):
        # Upper right: sample-to-target cross-sequence features
        upper_right = self._compute_cross_sequence_features(
            coords1="coords_nm", coords2="x_target"
        )  # [b, n_orig, n_concat, feature_dim]
        
        # Lower left: transpose of upper right (efficient!)
        lower_left = upper_right.transpose(1, 2)  # [b, n_concat, n_orig, feature_dim]
        
        # Lower right: target-to-target features  
        lower_right = self._compute_same_sequence_features(
            coords="x_target"
        )  # [b, n_concat, n_concat, feature_dim]
        
        # Assemble block matrix
        extended_pair_rep = torch.zeros(b, n_extended, n_extended, pair_dim)
        extended_pair_rep[:, :n_orig, :n_orig, :] = orig_pair_rep
        extended_pair_rep[:, :n_orig, n_orig:, :] = upper_right  
        extended_pair_rep[:, n_orig:, :n_orig, :] = lower_left
        extended_pair_rep[:, n_orig:, n_orig:, :] = lower_right
        
        return extended_pair_rep
```

### Cross-Sequence Feature Types

- **Sequence Separation**: Cross-sequence relative positions
- **Backbone Distances**: Pairwise backbone atom distances  
- **Chain Features**: Chain index relationships
- **Optional Features**: Conditional CA distances, self-conditioning features

## Configuration

### Basic Concat Features

```yaml
concat_features:
  enable_motif: true      # Enable motif concat features
  enable_target: true     # Enable target concat features  
  dim_feats_out: 768      # Output feature dimension
```

### Advanced Pair Features

```yaml
concat_features:
  enable_target: true
  target_pair_features: true    # Enable cross-sequence pair features
  dim_pair_out: 256            # Must match pair representation dimension
```

## Usage Examples

### 1. Motif-Conditioned Generation

```python
# Data pipeline
transforms = [
    MotifMaskTransform(motif_prob=0.8),
    ExtractMotifCoordinatesTransform(compact_mode=True),
    # ... other transforms
]

# Model configuration
config = {
    "concat_features": {
        "enable_motif": True,
        "dim_feats_out": 768
    }
}

# At inference, motif features are automatically included
model.forward(batch)  # Batch contains x_motif, motif_mask, seq_motif, etc.
```

### 2. Target-Conditioned Binder Design

```python
# Data pipeline for binder design
transforms = [
    TargetMaskTransform(selection_mode="random"),
    ExtractTargetCoordinatesTransform(compact_mode=True),
    FilterTargetResiduesTransform(),  # Remove targets from main features
    # ... other transforms
]

# Model with advanced pair features
config = {
    "concat_features": {
        "enable_target": True,
        "target_pair_features": True,
        "dim_feats_out": 768,
        "dim_pair_out": 256
    }
}
```

### 3. Dual Motif + Target Conditioning

```python
# Enable both motif and target conditioning
config = {
    "concat_features": {
        "enable_motif": True,
        "enable_target": True,
        "dim_feats_out": 768
    }
}

# Features from both motif and target are concatenated along sequence dimension
# Final extended sequence: [b, n_binder + n_motif + n_target, dim]
```

## Implementation Notes

### Memory Efficiency

- **Compact Mode**: Only motif/target residues stored, not full-length sparse arrays
- **Temporary Extension**: Sequence extended only during processing
- **Gradient Flow**: Direct gradients to motif/target features during training

### Coordinate Systems

- **Unit Consistency**: All coordinates in nanometers
- **Relative vs Absolute**: Both supported through feature configuration
- **Centering**: Compatible with various centering strategies

### Masking

- **Hierarchical Masking**: Residue-level masks control atom-level features
- **Padding Handling**: Automatic padding for variable-length sequences
- **Attention Masking**: Extended masks used for transformer attention

## Key Classes Summary

| Class | Purpose | Input | Output |
|-------|---------|-------|--------|
| `MotifConcatSeqFeat` | Extract motif features | Compact motif data | `[b, n_motif, 205]` |
| `TargetConcatSeqFeat` | Extract target features | Compact target data | `[b, n_target, 205]` |
| `ConcatFeaturesFactory` | Coordinate concat features | Multiple feature types | Extended sequence |
| `ConcatPairFeaturesFactory` | Cross-sequence pair features | Original pair rep | Extended pair rep |
| `LocalLatentsTransformer` | Neural network integration | Extended representations | Trimmed outputs |

The system provides a seamless way to condition protein generation models on structural motifs and target binding sites while maintaining computational efficiency and gradient flow. 