# Community Models

This directory contains external community model implementations that are used throughout the protein foundation models project. These packages have been consolidated here for better organization and maintainability.

## Included Packages

| Package | Description | Original Source |
|---------|-------------|-----------------|
| **colabdesign** | JAX-based protein design tools including AlphaFold2 integration, ProteinMPNN wrappers, and TrRosetta models | [sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign) |
| **openfold** | PyTorch reimplementation of AlphaFold2 with utilities for protein structure processing | [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold) |
| **ProteinMPNN** | Message Passing Neural Network for protein sequence design | [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) |
| **LigandMPNN** | Extension of ProteinMPNN with ligand and small molecule awareness | [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN) |
| **foundry** | RoseTTAFold3 implementation via IPSD's Foundry framework for structure prediction | [IPSD Foundry](https://github.com/baker-laboratory/RoseTTAFold3) |

## Foundry & RoseTTAFold3

**Foundry** is the Baker Lab's deep learning framework that powers RoseTTAFold3 (RF3). It provides a modular architecture for protein structure prediction with support for:

- **Multi-modal inputs**: Proteins, nucleic acids, small molecules, ions, and covalent modifications
- **Diffusion-based structure generation**: Uses denoising diffusion for coordinate prediction
- **Confidence estimation**: Built-in pLDDT and PAE-like metrics
- **Flexible inference**: Supports various input formats (FASTA, PDB, mmCIF)

### RF3 Overview

RoseTTAFold3 is a next-generation structure prediction model that extends RoseTTAFold2 with:

1. **All-atom modeling**: Predicts full atomic coordinates, not just backbone
2. **Ligand support**: Native handling of small molecules and cofactors
3. **Improved accuracy**: Better performance on protein-protein and protein-ligand complexes
4. **Faster inference**: Optimized architecture for practical use

### Using RF3

```python
# RF3 is integrated into the proteinfoundation pipeline
# Weights are stored at: community_models/ckpts/RF3/

# The checkpoint file rf3_foundry_01_24_latest_remapped.ckpt contains
# the pretrained RF3 model weights compatible with the local Foundry installation
```

## Directory Structure

```
community_models/
├── __init__.py
├── README.md
├── ckpts/                    # Model checkpoints
│   ├── AF2/                  # AlphaFold2 weights
│   ├── Boltz2/               # Boltz2 weights (optional)
│   └── RF3/                  # RoseTTAFold3 weights
├── colabdesign/              # JAX-based design tools
│   ├── af/                   # AlphaFold2 modules
│   ├── mpnn/                 # ProteinMPNN JAX wrapper
│   ├── tr/                   # TrRosetta modules
│   ├── rf/                   # RoseTTAFold modules
│   ├── shared/               # Shared utilities
│   └── esm_msa/              # ESM-MSA modules
├── foundry/                  # RF3 Foundry framework
│   ├── src/foundry/          # Core implementation
│   ├── models/               # Model definitions
│   └── examples/             # Usage examples
├── openfold/                 # PyTorch AlphaFold2 implementation
│   ├── model/                # Model architecture
│   ├── data/                 # Data processing
│   ├── np/                   # NumPy utilities & residue constants
│   └── utils/                # General utilities
├── ProteinMPNN/              # Original ProteinMPNN
│   ├── protein_mpnn_run.py
│   ├── protein_mpnn_utils.py
│   ├── ca_model_weights/
│   └── vanilla_model_weights/
└── LigandMPNN/               # Ligand-aware MPNN
    ├── run.py
    ├── score.py
    ├── model_params/
    └── openfold/             # Local openfold copy (see notes below)
```

## Import System

Thanks to the `pyproject.toml` configuration, all community packages are exposed as **top-level imports**. This means:

- **No import changes required** when updating from upstream
- **Original imports work as-is** within each package
- **Both import styles work** for external usage

```python
# Both of these work:
from openfold.np import residue_constants
from community_models.openfold.np import residue_constants

# Original package imports work unchanged:
from colabdesign import mk_afdesign_model
from openfold.np.protein import from_pdb_string
```

This is achieved via the `[tool.hatch.build.targets.wheel.sources]` configuration in `pyproject.toml`, which maps the subpackages to their expected import paths.

### LigandMPNN Local openfold

LigandMPNN contains its own copy of openfold in `LigandMPNN/openfold/`. This is intentional and ensures LigandMPNN remains self-contained, avoiding conflicts with the main openfold package.

## Usage

### Importing Packages

```python
# OpenFold utilities (both styles work)
from openfold.np import residue_constants
from openfold.np.protein import from_pdb_string

# ColabDesign (requires JAX environment)
from colabdesign import mk_afdesign_model, mk_mpnn_model

# ProteinMPNN utilities
from ProteinMPNN import protein_mpnn_utils
```

### Running Command-Line Tools

```bash
# ProteinMPNN
python ./community_models/ProteinMPNN/protein_mpnn_run.py \
    --pdb_path input.pdb \
    --out_folder output/

# LigandMPNN  
python ./community_models/LigandMPNN/run.py \
    --pdb_path input.pdb \
    --out_folder output/ \
    --model_type ligand_mpnn
```

## Model Weights

Model weights are organized across two directories:

- **`ckpts/`** (project root) - Complexa model checkpoints
- **`community_models/ckpts/`** - Structure prediction models (AF2, Boltz2, RF3)

| Model | Location | Size | Notes |
|-------|----------|------|-------|
| **Complexa** | `ckpts/complexa.ckpt` | TBD | Flow matching model (required) |
| **Complexa AE** | `ckpts/complexa_ae.ckpt` | TBD | Autoencoder (required) |
| **ProteinMPNN** | `community_models/ProteinMPNN/ca_model_weights/` | ~50MB | CA-only models |
| **ProteinMPNN** | `community_models/ProteinMPNN/vanilla_model_weights/` | ~50MB | Full backbone models |
| **LigandMPNN** | `community_models/LigandMPNN/model_params/` | ~500MB | Ligand-aware variants |
| **AlphaFold2** | `community_models/ckpts/AF2/` | ~5GB | All model variants (1-5, ptm, multimer) |
| **RoseTTAFold3** | `community_models/ckpts/RF3/` | ~2.5GB | Optional - multi-modal prediction |

### Downloading Weights

Use the interactive download wizard:

```bash
bash env/download_startup.sh
```

Or download specific models via CLI:

```bash
# Complexa model (required)
bash env/download_startup.sh --complexa

# Core models (ProteinMPNN + LigandMPNN + AF2)
bash env/download_startup.sh --all

# Individual models
bash env/download_startup.sh --pmpnn
bash env/download_startup.sh --ligmpnn
bash env/download_startup.sh --af2

# Optional models
bash env/download_startup.sh --rf3

# Everything (core + optional + complexa)
bash env/download_startup.sh --everything
```

The download script automatically skips files that are already present.

Complexa model checkpoints are hosted on [NGC](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/proteina_complexa) and downloaded automatically by the script.

## Maintenance Notes

### Updating Upstream Packages

When updating from upstream repositories:

1. **No import changes needed**: The `pyproject.toml` source mapping handles path resolution automatically
2. **LigandMPNN's openfold**: LigandMPNN has its own openfold copy - updates to the main openfold don't affect it (and vice versa)
3. **ColabDesign dependencies**: Requires JAX ecosystem (`jax`, `jaxlib`, `haiku`, `optax`)
4. **OpenFold dependencies**: Requires PyTorch and `dm-tree`
5. **Foundry dependencies**: Requires PyTorch, specific CUDA versions for optimal performance

### Adding New Community Models

To add a new community model:

1. Place the package directory under `community_models/`
2. Create an `__init__.py` if not present
3. Add the package to `pyproject.toml` sources mapping (if top-level imports are desired)
4. Add weights download logic to `env/download_startup.sh` if applicable
5. Document in this README

## License

Each package retains its original license:

- **colabdesign**: MIT License
- **openfold**: Apache 2.0 License  
- **ProteinMPNN**: MIT License
- **LigandMPNN**: MIT License
- **foundry**: BSD 3-Clause License

See individual package directories for full license texts.
