# Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute (ICLR 2026 Oral Paper)

<div align="center">
  <a href="https://kdidi.netlify.app/" target="_blank">Kieran&nbsp;Didi</a><sup>*</sup> &emsp;
  <a href="https://oxer11.github.io/" target="_blank">Zuobai&nbsp;Zhang</a><sup>*</sup> &emsp;
  <a href="https://scholar.google.com/citations?user=CsfbiNUAAAAJ&hl=en" target="_blank">Guoqing&nbsp;Zhou</a><sup>*</sup> &emsp;
  <a href="https://scholar.google.com/citations?user=KBn52kYAAAAJ&hl=en" target="_blank">Danny&nbsp;Reidenbach</a><sup>*</sup>
  <br>
  <a href="https://scholar.google.com/citations?user=wGjVFHIAAAAJ&hl=en" target="_blank">Zhonglin&nbsp;Cao</a><sup>*</sup> &emsp;
  <a href="https://steineggerlab.com/en/authors/sooyoung/" target="_blank">Sooyoung&nbsp;Cha</a><sup>*</sup> &emsp;
  <a href="https://tomasgeffner.github.io/" target="_blank">Tomas&nbsp;Geffner</a> &emsp;
  <a href="https://christian.dallago.us/" target="_blank">Christian&nbsp;Dallago</a>
  <br>
  <a href="https://jian-tang.com/" target="_blank">Jian&nbsp;Tang</a> &emsp;
  <a href="https://www.cs.ox.ac.uk/people/michael.bronstein/" target="_blank">Michael&nbsp;M.&nbsp;Bronstein</a> &emsp;
  <a href="https://steineggerlab.com/en/authors/admin/" target="_blank">Martin&nbsp;Steinegger</a>
  <br>
  <a href="https://scholar.google.ch/citations?user=LUXL9FoAAAAJ&hl=en" target="_blank">Emine&nbsp;Kucukbenli</a><sup>&loz;</sup> &emsp;
  <a href="http://arashvahdat.com/" target="_blank">Arash&nbsp;Vahdat</a><sup>&loz;</sup> &emsp;
  <a href="https://karstenkreis.github.io/" target="_blank">Karsten&nbsp;Kreis</a><sup>&dagger;</sup>
  <br> <br>
  <!-- <sub>
    <sup>1</sup>NVIDIA &ensp;
    <sup>2</sup>University of Oxford &ensp;
    <sup>3</sup>Mila - Qu&eacute;bec AI Institute &ensp;
    <sup>4</sup>Universit&eacute; de Montr&eacute;al &ensp;
    <sup>5</sup>HEC Montr&eacute;al
    <br>
    <sup>6</sup>CIFAR AI Chair &ensp;
    <sup>7</sup>AITHYRA &ensp;
    <sup>8</sup>School of Biological Sciences, Seoul National University
    <br>
    <sup>9</sup>Interdisciplinary Program in Bioinformatics, Seoul National University &ensp;
    <sup>10</sup>Institute of Molecular Biology and Genetics, Seoul National University
    <br>
    <sup>11</sup>Artificial Intelligence Institute, Seoul National University
  </sub>
  <br> <br> -->
  <span><sup>*</sup>Core contributor. &emsp; <sup>&loz;</sup>Equal advising. &emsp; <sup>&dagger;</sup>Project lead.</span>
  <br> <br>
  <a href="https://openreview.net/forum?id=qmCpJtFZra" target="_blank">Paper</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://research.nvidia.com/labs/genair/proteina-complexa/" target="_blank">Project&nbsp;Page</a>
</div>

<br>
<br>

<div align="center">
    <img width="600" alt="teaser" src="assets/pipeline_figure.png"/>
</div>

<br>
<br>

*Abstract.* Protein interaction modeling is central to protein design, which has been transformed by machine learning with applications in drug discovery and beyond. In this landscape, structure-based de novo binder design is cast as either conditional generative modeling or sequence optimization via structure predictors ("hallucination"). We argue that this is a false dichotomy and propose Proteina-Complexa, a novel fully atomistic binder generation method unifying both paradigms. We extend recent flow-based latent protein generation architectures and leverage the domain-domain interactions of monomeric computationally predicted protein structures to construct Teddymer, a new large-scale dataset of synthetic binder-target pairs for pretraining. Combined with high-quality experimental multimers, this enables training a strong base model. We then perform inference-time optimization with this generative prior, unifying the strengths of previously distinct generative and hallucination methods. Proteina-Complexa sets a new state of the art in computational binder design benchmarks: it delivers markedly higher in-silico success rates than existing generative approaches, and our novel test-time optimization strategies greatly outperform previous hallucination methods under normalized compute budgets. We also demonstrate interface hydrogen bond optimization, fold class-guided binder generation, and extensions to small molecule targets and enzyme design tasks, again surpassing prior methods. Code, models and new data will be publicly released.

Find the Model Card++ for Proteina-Complexa [here](./assets/model_card/overview.md).

---

## Overview

Proteina-Complexa is a generative model for protein complex design using flow matching. It enables the design of protein binders through a unified framework that models backbone geometry, side-chain conformations, and sequences jointly.

**Key Capabilities:**
- **Protein binder design** — Design novel protein binders for target proteins
- **Ligand binder design** — Design protein binders for small-molecule ligands
- **Motif scaffolding (AME)** — Scaffold functional motifs with ligand context using the AME model
- **Search-based optimization** — Reward models (AlphaFold2, RoseTTAFold3, force fields)
- **Integrated sequence design** — Self (no re-design) / ProteinMPNN / SoluableMPNN / LigandMPNN
- **Structure prediction validation** — AlphaFold2, ESMFold, and RoseTTAFold3
- **Comprehensive evaluation** — Binder refolding, monomer designability, motif RMSD, and combined motif-binder metrics
- **Automated analysis** — Structured result CSVs, success filtering, and diversity computation

### Wet-Lab Validation

Proteina-Complexa designs have been experimentally validated across diverse protein targets and interaction types, demonstrating that in-silico success translates to real binding activity. For full experimental results, protocols, and characterization data, see the [paper](https://research.nvidia.com/labs/genair/proteina-complexa/assets/proteina_complexa_validation/proteina_complexa_validation.pdf) and [project website](https://research.nvidia.com/labs/genair/proteina-complexa/).

## Installation

### Option 1: UV Environment (Recommended)

> **Note**: Requires Ubuntu 22.04+ or equivalent. Ubuntu 20.04 will throw GLIBC errors due to older system libraries. Use Docker (Option 2) for older systems.

```bash
git clone https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa
cd Proteina-Complexa

./env/build_uv_env.sh

# Activate the environment
source .venv/bin/activate
```

> **Known Issue: tmol install fails on Python 3.12 for some users**
>
> tmol depends on `sparse` -> `numba` -> `llvmlite`, and the pinned versions of `llvmlite` are
> incompatible with Python 3.12. If the install fails during the tmol step, add the following
> workaround to `build_uv_env.sh` before the tmol install line:
>
> ```bash
> if [[ "$PYTHON_VERSION" == "3.12" ]]; then
>     # Pre-install 3.12-compatible versions of llvmlite/numba
>     uv pip install "llvmlite>=0.41" "numba>=0.59" || true
> fi
> uv pip install "git+https://github.com/uw-ipd/tmol.git" || echo "Warning: tmol install failed"
> ```
>
> This is not needed on all systems -- it depends on which versions of `llvmlite` and `numba`
> your resolver picks up. If the default install works for you, no action is needed.

### Option 2: Docker Container

```bash
# Build the image (includes UV env, Foldseek, MMseqs2, DSSP, SC)
docker build -t proteina-complexa -f env/docker/Dockerfile .

# Run interactively with GPU access
docker run --gpus all -it proteina-complexa

# Or mount a local data directory
docker run --gpus all -it \
    -v /path/to/PFM_data:/workspace/data \
    proteina-complexa
```

The container activates the UV environment automatically. AF2, RF3, DSSP, and SC paths are pre-configured. See the [Dockerfile](env/docker/Dockerfile) for the full list of environment variables.

### Post-Installation Setup

```bash
# Initialize environment configuration
complexa init

# Download required model weights
complexa download
```

For **evaluation** (e.g. RF3 refolding or RF3 reward during generation), set `RF3_CKPT_PATH` and `RF3_EXEC_PATH` in your environment or `.env`. See [Evaluation & Analysis Guide](docs/EVALUATION_METRICS.md#environment-variables) for the full list of environment variables.

## Configuration

Complexa uses [Hydra](https://hydra.cc/) + [OmegaConf](https://omegaconf.readthedocs.io/) for configuration. There are two categories of paths you need to set up:

1. **Model checkpoint paths** — set directly in the pipeline YAML configs
2. **Community models & bioinformatics tools** — set as environment variables in `.env`

### Step 1: Set Model Checkpoint Paths

Each pipeline config has three checkpoint fields at the top level that you **must** point to your local checkpoint files:

```yaml
ckpt_path: /path/to/your/checkpoints       # Directory containing the model checkpoint
ckpt_name: complexa.ckpt                    # Checkpoint filename
autoencoder_ckpt_path: /path/to/your/checkpoints/complexa_ae.ckpt  # Full path to autoencoder checkpoint
```

There are **three model variants**, each with its own checkpoint pair (model + autoencoder), hosted on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/proteina_complexa):

| Pipeline | Config File | Checkpoint Fields | NGC |
|----------|------------|-------------------|-----|
| **Protein Binder** | `configs/search_binder_local_pipeline.yaml` | `complexa.ckpt` + `complexa_ae.ckpt` | [proteina_complexa](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/proteina_complexa) |
| **Ligand Binder** | `configs/search_ligand_binder_local_pipeline.yaml` | `complexa_ligand.ckpt` + `complexa_ligand_ae.ckpt` | [proteina_complexa_ligand](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/proteina_complexa_ligand) |
| **AME (Motif Scaffolding)** | `configs/search_ame_local_pipeline.yaml` | `complexa_ame.ckpt` + `complexa_ame_ae.ckpt` | [proteina_complexa_ame](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/proteina_complexa_ame) |

Download checkpoints with `complexa download --complexa-all` (see [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/proteina_complexa) for details).

Edit the pipeline config you plan to use and set the paths to where you downloaded the checkpoints:

```bash
# Example: edit the protein binder config
# Change these three lines to match your local paths:
#   ckpt_path: /your/checkpoint/directory
#   ckpt_name: complexa.ckpt
#   autoencoder_ckpt_path: /your/checkpoint/directory/complexa_ae.ckpt
```

You can also override checkpoint paths from the command line without editing the YAML:

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++ckpt_path=/my/checkpoints \
    ++ckpt_name=complexa.ckpt \
    ++autoencoder_ckpt_path=/my/checkpoints/complexa_ae.ckpt
```

> **Verify your checkpoints exist** before running. Missing or incorrect paths will cause immediate failures at model loading time.

### Step 2: Set Environment Variables for Community Models & Tools

Community models (oracle/reward models) and bioinformatics tools are referenced in configs via `${oc.env:VARIABLE_NAME}`, which resolves environment variables at runtime. These are loaded from your `.env` file (created by `complexa init`).

Edit `.env` and set the paths for any models or tools you have installed:

```bash
# ── Community Model Checkpoints ──
AF2_DIR=/path/to/community_models/ckpts/AF2           # AlphaFold2 parameters
ESM_DIR=/path/to/community_models/ckpts/ESM2          # ESM2 weights
RF3_CKPT_PATH=/path/to/rf3_checkpoint.ckpt            # RoseTTAFold3 checkpoint
RF3_EXEC_PATH=/path/to/.venv/bin/rf3                  # RoseTTAFold3 executable

# ── Bioinformatics Tool Binaries ──
SC_EXEC=/path/to/sc                                    # CCP4 shape complementarity (see note below)
FOLDSEEK_EXEC=/path/to/foldseek                        # Foldseek structural search
MMSEQS_EXEC=/path/to/mmseqs                            # MMseqs2 sequence clustering
DSSP_EXEC=/path/to/dssp                                # DSSP secondary structure (see note below)

# ── Paths ──
LOCAL_CODE_PATH=/path/to/Proteina-Complexa     # Project root
COMMUNITY_MODELS_PATH=${LOCAL_CODE_PATH}/community_models
DATA_PATH=/path/to/PFM_data                            # Target structures and datasets
```

> **`dssp` and `sc` binaries**: These are not distributed with this repository. Pre-compiled binaries compatible with our evaluation pipeline can be obtained from [FreeBindCraft](https://github.com/cytokineking/FreeBindCraft/tree/master/functions) (`dssp` and `sc`). Place them in `env/docker/internal/` for Docker builds, or anywhere on your system and set `DSSP_EXEC` and `SC_EXEC` in `.env` accordingly. The Docker image includes them automatically from `env/docker/internal/`.

> **Full `.env` reference**: The `.env` file also contains Docker settings, SLURM cluster configuration, and runtime-specific tool paths.

The reward model configs in `configs/pipeline/binder/binder_generate.yaml` (and the ligand/AME equivalents) reference these variables:

```yaml
# AF2 reward model — resolves AF2_DIR from .env
af2folding:
  _target_: "proteinfoundation.rewards.alphafold2_reward.AF2RewardModel"
  af_params_dir: ${oc.env:AF2_DIR}

# RF3 reward model — resolves RF3_CKPT_PATH and RF3_EXEC_PATH from .env
rf3folding:
  _target_: "proteinfoundation.rewards.rf3_reward.RF3RewardRunner"
  ckpt_path: ${oc.env:RF3_CKPT_PATH}
  rf3_path: ${oc.env:RF3_EXEC_PATH}
```

If an environment variable is missing, Hydra will fail with an `InterpolationKeyError` at config resolution time — this is intentional so you know exactly what is missing.

### Verifying Your Setup

After configuring checkpoint paths and environment variables, verify everything is accessible:

```bash
# Validate the config resolves without errors
complexa validate design configs/search_binder_local_pipeline.yaml
```

> **Detailed config reference**: See [Configuration Guide](docs/CONFIGURATION_GUIDE.md) for all YAML fields — search algorithms, reward model weights, evaluation settings, analysis thresholds, and training configs.

## Quick Start

> **GPU parallelism**: The pipeline configs default to `gen_njobs: 1` and `eval_njobs: 1`, which uses a single GPU. If you have multiple GPUs, increase these to run generation and evaluation in parallel — each job uses one GPU. For example, with 4 GPUs:
>
> ```bash
> complexa design configs/search_binder_local_pipeline.yaml \
>     ++gen_njobs=4 ++eval_njobs=4 ...
> ```
>
> Or edit `gen_njobs` / `eval_njobs` directly in the pipeline YAML.

```bash
# 1. Setup
source .venv/bin/activate
complexa init
complexa download --all

# 2. Validate configuration
complexa validate design configs/search_binder_local_pipeline.yaml

# 3. Design binders for PDL1
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=pdl1_test \
    ++generation.task_name=02_PDL1

# 4. Check results
complexa status configs/search_binder_local_pipeline.yaml
```

Other pipeline types:

```bash
# Ligand binder design
complexa design configs/search_ligand_binder_local_pipeline.yaml \
    ++run_name=ligand_test \
    ++generation.task_name=39_7V11_LIGAND

# AME motif + ligand binder scaffolding
complexa design configs/search_ame_local_pipeline.yaml \
    ++run_name=ame_test \
    ++generation.task_name=M0024_1nzy_v3

# Monomer motif scaffolding (indexed mode). Note motif targets not provided
complexa design configs/search_motif_local_pipeline.yaml \
    ++run_name=motif_test \
    ++generation.task_name=1YCR_AA
```

> **Known limitation: TMOL reward not supported for ligand binder / AME pipelines**
>
> The TMOL force-field reward model currently does not work with protein-ligand complexes. The TMOL reward section is commented out by default in the ligand binder and AME generate configs. If you enable it, TMOL scores will be fail and only the other reward models (e.g. RF3) will contribute to the reward.

### Preparing AME Input Structures

> **Warning**: AME input PDBs must follow a specific chain and naming convention. Malformed inputs will cause silent failures or incorrect results.
>
> - **Ligand(s)** must be on **chain A** with residue name set to `L:0`
> - **Motif protein residues** must be on **chain B**
> - If your source PDB has different chain assignments, re-chain and rename before adding the entry to `configs/design_tasks/ame_dict_v2.yaml`
>
> The bundled targets in `assets/target_data/ame_input_structures/` are not all already prepared this way. M0024_1nzy_v3 is provided as the example. If you add custom targets, follow the same convention. See [`assets/target_data/README.md`](assets/target_data/README.md) for full preparation instructions.

### Evaluating AME Designs with Ligand Targets (RF3)

When running RF3 evaluation on AME-generated PDB files that contain a ligand (small molecule on chain A), RF3 will attempt to add missing atoms based on the ligand's CCD (Chemical Component Dictionary) code. This can cause shape errors in downstream RMSD calculations as well as provide the incorrect structure.

**Solution:** If not already done in the conditioning input, replace the ligand residue name with `L:0` before passing to RF3. This tells RF3 to treat it as a generic ligand and skip atom completion.

```python
from atomworks.io import load_any, to_pdb_file

atom_array = load_any("my_design.pdb")[0]

# Select chain A (ligand) and rename residues
ligand_mask = atom_array.chain_id == "A"
atom_array.res_name[ligand_mask] = "L:0"

to_pdb_file(atom_array, "my_design_rf3_ready.pdb")
```

## CLI Reference

The `complexa` CLI provides a unified interface for all operations.

```bash
complexa --help      # Show all commands
```

| Command | Description |
|---------|-------------|
| `complexa init` | Initialize environment configuration (.env file) |
| `complexa download` | Download model weights (interactive wizard) |
| `complexa validate` | Validate configuration before running |
| `complexa design` | Run full pipeline: generate → filter → evaluate → analyze |
| `complexa generate` | Generate binder structures |
| `complexa filter` | Filter samples by reward scores |
| `complexa evaluate` | Evaluate with structure prediction |
| `complexa analyze` | Aggregate and analyze results |
| `complexa analysis` | Run evaluate → analyze pipeline (for evaluating PDB files) |
| `complexa target` | Target management (list, add, show) |
| `complexa status` | Check pipeline status and outputs |
| `complexa demo` | Show usage examples and explanations |

### Design Pipelines

There are four design pipelines, each with its own config and model:

| Pipeline | Config | Model | Use Case |
|----------|--------|-------|----------|
| Protein Binder | `search_binder_local_pipeline.yaml` | Protein model | Design binders for protein targets |
| Ligand Binder | `search_ligand_binder_local_pipeline.yaml` | Ligand model (LoRA) | Design binders for small-molecule ligands |
| AME (Motif + Ligand) | `search_ame_local_pipeline.yaml` | AME model (LoRA) | Scaffold functional motifs with ligand context |
| Monomer Motif | `search_motif_local_pipeline.yaml` | AME model (LoRA) | Scaffold structural motifs into monomer proteins |

Each pipeline runs four stages: **generate → filter → evaluate → analyze**. Run the full pipeline with `complexa design`, or run stages individually (`complexa generate`, `complexa filter`, etc.). See the [Inference Guide](docs/INFERENCE.md) for individual stages, SLURM deployment, and advanced usage.

## Data

Complexa models are trained on protein structure data from several public sources.

### Protein Data Bank (PDB)

The [RCSB Protein Data Bank](https://www.rcsb.org/) is the primary source of experimentally determined protein structures used for model fine-tuning. Target structures for binder design tasks (in `assets/target_data/`) are derived from PDB entries.

The 45,856 PDB multimer IDs used for training are listed in [`assets/data/pdb_multimer_ids.txt`](assets/data/pdb_multimer_ids.txt). The full metadata CSV ([`assets/data/pdb_multimer.csv`](assets/data/pdb_multimer.csv)) contains the following columns:

| Column | Description |
|--------|-------------|
| `pdb` | PDB accession code (e.g. `104l`) |
| `id` | Per-chain identifiers (e.g. `['104l_A', '104l_B']`) |
| `chain` | Chain letters |
| `length` | Residue count per chain |
| `molecule_type` | Molecule type per chain (e.g. `protein`) |
| `name` | Molecule name per chain |
| `sequence` | Amino acid sequence per chain |
| `split` | Dataset split assignment per chain |
| `n_chains` | Number of chains in the complex |
| `ligands` | Ligand identifiers per chain (CCD codes) |
| `source` | Source organism per chain |
| `resolution` | Experimental resolution (Å) per chain |
| `deposition_date` | PDB deposition date per chain |
| `experiment_type` | Experimental method per chain (e.g. `diffraction`) |
| `pdb_file_available` | Whether the PDB file is available per chain |
| `n_available_chains` | Number of chains with available structures |
| `total_length` | Total residue count across all chains |

### PLINDER

[PLINDER](https://www.plinder.sh/) provides curated protein-ligand interaction data at scale, with paired unbound and predicted structures. This is used for ligand binder and AME model training.

The complete PLINDER system IDs used listed in [`assets/data/plinder_valid_ids.txt`](assets/data/plinder_valid_ids.txt). Each ID follows the PLINDER naming convention (e.g. `4rek__1__1.A__1.B`).

The training dataloader ([`configs/dataset/unified/plinder.yaml`](configs/dataset/unified/plinder.yaml)) expects `metadata_file` to point to a CSV with the following columns:

| Column | Description |
|--------|-------------|
| `complex_name` | PLINDER system identifier (e.g. `4rek__1__1.A__1.B`) |
| `protein_path` | Relative path to the receptor PDB file |
| `ligand_paths` | Relative path to the ligand SDF file |
| `num_residues_protein` | Number of protein residues in the complex |
| `num_heavy_atoms_ligand` | Number of heavy atoms in the ligand |
| `path` | Absolute path to the system `.cif` structure file (used by the dataloader via `path_column`) |
| `example_id` | Unique sample identifier (used by the dataloader via `id_column`) |

The dataloader requires at minimum `example_id` (the ID column) and `path` (the path column). Additional columns to load are specified in the config's `columns_to_load` field.

### Teddymer

[Teddymer](http://teddymer.foldseek.com/) provides non-singleton structural clusters generated by Foldseek-Multimer-Clustering over the TED (domain-segmented AlphaFold) database. These multimer structures are used as training data for the binder model, providing diverse protein-protein interface geometries at scale.

The Teddymer dimer IDs used for training are listed in [`assets/data/teddymer_valid_ids.txt`](assets/data/teddymer_valid_ids.txt). Each ID follows the format `DI{cluster_id}_AF-{uniprot_id}-F1-model_v4` (e.g. `DI655_AF-A0A009ESU5-F1-model_v4`).

## Documentation

| Document | What it covers |
|----------|---------------|
| [Configuration Guide](docs/CONFIGURATION_GUIDE.md) | All YAML config examples: search, rewards, evaluation, analysis, training |
| [Inference Guide](docs/INFERENCE.md) | Running locally and on SLURM, custom targets, troubleshooting |
| [Search Metadata & Visualization](docs/SEARCH_METADATA.md) | Metadata tag conventions, output size formulas, trajectory query examples |
| [Evaluation & Analysis Guide](docs/EVALUATION_METRICS.md) | Evaluation pipeline, metrics, result CSVs, success criteria |
| [Sweep System](docs/SWEEP.md) | Parameter sweeps without modifying source code |
| [Training Guide](docs/TRAINING.md) | Model training instructions |

## Evaluation Types

Each pipeline automatically runs the appropriate evaluation. Six evaluation types cover different design tasks -- from standard protein binders to joint motif-binder assessments with ligand clash detection. See the [Evaluation & Analysis Guide](docs/EVALUATION_METRICS.md#protein-types--result-types) for the full type mapping, default thresholds, and success criteria.

## Project Structure

```
Proteina-Complexa/
├── src/proteinfoundation/           # Main package
│   ├── cli/                         # Command-line interface (complexa CLI)
│   ├── datasets/                    # Data loading, transforms, atomworks integration
│   ├── flow_matching/               # Flow matching implementation
│   ├── metrics/                     # Metric computation utilities
│   ├── nn/                          # Neural network architectures
│   ├── partial_autoencoder/         # Partial autoencoder for latent representations
│   ├── rewards/                     # Reward models (AF2, RF3, force fields)
│   ├── search/                      # Search algorithms (beam, MCTS, best-of-N, FK steering)
│   ├── evaluation/                  # Per-sample evaluation (evaluate step)
│   │   ├── binder_eval.py           #   Binder refolding & binding metrics
│   │   ├── monomer_eval.py          #   Monomer designability & codesignability
│   │   ├── motif_eval.py            #   Motif RMSD, sequence recovery
│   │   └── motif_binder_eval.py     #   Joint motif + binder evaluation
│   ├── result_analysis/             # Aggregate analysis (analyze step)
│   │   ├── binder_analysis.py       #   Binder success rates & filtering
│   │   ├── monomer_analysis.py      #   Monomer pass rates & filtering
│   │   ├── motif_analysis.py        #   Motif success criteria & filtering
│   │   ├── motif_binder_analysis.py #   Joint motif-binder analysis
│   │   └── compute_diversity.py     #   FoldSeek & MMseqs diversity
│   ├── utils/                       # Shared utilities (PDB, alignment, EMA, etc.)
│   ├── proteina.py                  # Core model (Lightning module)
│   ├── train.py                     # Training entry point
│   ├── generate.py                  # Generation entry point
│   ├── filter.py                    # Filtering entry point
│   ├── evaluate.py                  # Evaluation entry point
│   └── analyze.py                   # Analysis entry point
├── configs/                         # Hydra configuration files
│   ├── search_binder_local_pipeline.yaml         # Protein binder pipeline (local)
│   ├── search_ligand_binder_local_pipeline.yaml  # Ligand binder pipeline (local)
│   ├── search_ame_local_pipeline.yaml            # AME motif scaffolding pipeline (local)
│   ├── search_binder_pipeline.yaml               # Protein binder pipeline (SLURM)
│   ├── evaluate*.yaml               # Standalone evaluation configs
│   ├── analyze*.yaml                # Standalone analysis configs
│   ├── training*.yaml               # Training configs
│   ├── pipeline/                    # Modular stage configs
│   │   ├── binder/                  #   Protein binder (generate, evaluate, analyze + SLURM variants)
│   │   ├── ligand_binder/           #   Ligand binder (generate, evaluate, analyze)
│   │   ├── ame/                     #   AME motif scaffolding (generate, evaluate, analyze)
│   │   ├── model_sampling.yaml      #   Shared diffusion sampling params
│   │   └── base_gen_data.yaml       #   Shared data loader config
│   ├── dataset/                     # Dataset configs
│   ├── nn/                          # Model architecture configs
│   ├── generation/                  # Generation parameter configs
│   └── targets/                     # Target protein/ligand definitions
├── docs/                            # Documentation
├── env/                             # Environment setup (UV, Docker)
└── slurm_utils/                     # SLURM cluster launch scripts
```

## Citation

```bibtex
@article{didi2026invitro,
  title={Latent Generative Search Unlocks de novo Design of Untapped Biomolecular Interactions at Scale},
  author={Kieran Didi and Danny Reidenbach and Matthew Penner and Supriya Ravi and Marshall Case and Mike Nichols and Erik Swanson and Alex Reis and Maggie Prescott and Yue Qian and Dongming Qian and Jingjing Yang and Weiji Li and Le Li and Daichi Shonai and Sean Gay and Bhoomika Basu Mallik and Ho Yeung Chim and Liurong Chen and Miguel Atienza Juantay and Hubert Klein and Anna Macintyre and Maxim Secor and Daniele Granata and Zhonglin Cao and Guoqing Zhou and Tomas Geffner and Xi Chen and Micha Livne and Zuobai Zhang and Tianjing Zhang and Michael M. Bronstein and Martin Steinegger and Kristine Deibler and Scott Soderling and Alena Khmelinskaia and Florian Hollfelder and Christian Dallago and Emine Kucukbenli and Arash Vahdat and Pierce Ogden and Karsten Kreis},
  year={2026}
}

@inproceedings{didi2026scaling,
  title={Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute},
  author={Kieran Didi and Zuobai Zhang and Guoqing Zhou and Danny Reidenbach and Zhonglin Cao and Sooyoung Cha and Tomas Geffner and Christian Dallago and Jian Tang and Michael M. Bronstein and Martin Steinegger and Emine Kucukbenli and Arash Vahdat and Karsten Kreis},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=qmCpJtFZra}
}

@inproceedings{geffner2026laproteina,
  title={La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching},
  author={Tomas Geffner and Kieran Didi and Zhonglin Cao and Danny Reidenbach and Zuobai Zhang and Christian Dallago and Emine Kucukbenli and Karsten Kreis and Arash Vahdat},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=RDerF20JYT}
}

@inproceedings{geffner2025proteina,
  title={Proteina: Scaling Flow-based Protein Structure Generative Models},
  author={Geffner, Tomas and Didi, Kieran and Zhang, Zuobai and Reidenbach, Danny and Cao, Zhonglin and Yim, Jason and Geiger, Mario and Dallago, Christian and Kucukbenli, Emine and Vahdat, Arash and Kreis, Karsten},
  booktitle={ICLR},
  year={2025}
}
```

## License

See [LICENSE](LICENSE) for details.
