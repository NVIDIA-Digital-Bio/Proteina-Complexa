# Monomer Motif Scaffolding Configs

Two modes for motif scaffolding — **indexed** and **unindexed** — each with
a paired generate + evaluate config. A shared analyze config handles both.

## Quick reference

| File | Purpose |
|------|---------|
| `idx_motif_generate.yaml` | Indexed generation — motif placed at contig positions |
| `idx_motif_evaluate.yaml` | Indexed evaluation — contig-based motif alignment |
| `uidx_motif_generate.yaml` | Unindexed generation — motif without positional encoding |
| `uidx_motif_evaluate.yaml` | Unindexed evaluation — greedy coordinate matching |
| `motif_analyze.yaml` | Shared analysis — pass rates, diversity (both modes) |

## Indexed vs Unindexed

**Indexed** (`padding=True`): The motif is embedded at specific contig
positions. `MotifFeatures` owns the dataset size — set `nsamples: 100` inside
the feature. `nres` must be `null`.

**Unindexed** (`padding=False`): The model receives only the bare motif atoms.
Scaffold length is controlled by `UniformInt` via `nres`. `MotifFeatures`
defaults to `nsamples=1` (one placement, stripped and replicated for all
samples).

## Running a pipeline

The top-level pipeline config composes generate + evaluate + analyze.
Edit `configs/search_motif_local_pipeline.yaml` to point at the correct
mode:

```yaml
# --- Indexed ---
defaults:
  - pipeline/motif/idx_motif_generate@generation
  - pipeline/motif/idx_motif_evaluate@_global_
  - pipeline/motif/motif_analyze@_global_
  - _self_

# --- OR Unindexed ---
defaults:
  - pipeline/motif/uidx_motif_generate@generation
  - pipeline/motif/uidx_motif_evaluate@_global_
  - pipeline/motif/motif_analyze@_global_
  - _self_
```

Then run exactly like the binder pipeline:

```bash
complexa design configs/search_motif_local_pipeline.yaml \
  ++run_name=motif_1ycr \
  ++generation.task_name=1YCR_AA
```

## What to change vs the binder pipeline

| Setting | Binder | Motif |
|---------|--------|-------|
| Checkpoint | `complexa.ckpt` | `complexa_ame.ckpt` |
| `env_vars.USE_V2_COMPLEXA_ARCH` | `"False"` | `"False"` |
| LoRA | not set | `r: 32, lora_alpha: 64.0` |
| `generation.task_name` | target name (e.g. `38_TNFalpha`) | motif task (e.g. `1YCR_AA`) |

## Changing the motif task

Override on the command line:

```bash
++generation.task_name=5TPN_AA       # all-atom
++generation.task_name=1YCR_AA_TIP   # tip-atoms
++generation.task_name=7MRX_AA_85    # multi-size variant
```

Tasks are defined in `configs/design_tasks/motif_dict.yaml`.

## Adjusting sample count

```bash
# Indexed — MotifFeatures controls dataset size
++generation.dataloader.dataset.conditional_features.0.nsamples=200

# Unindexed — UniformInt controls dataset size
++generation.dataloader.dataset.nres.nsamples=200
```

## Sweeping over multiple motif tasks

Create a sweep YAML in `configs/sweeps/` and use the same sweep tooling as
binders. The sweep system computes the cartesian product of all axes.

**1. Create a sweep file** (e.g. `configs/sweeps/motif_tasks.yaml`):

```yaml
# Sweep over multiple motif tasks
generation.task_name:
  - 1YCR_AA
  - 5TPN_AA
  - 1BCF_AA
  - 5IUS_AA
  - 3IXT_AA
```

**2. Generate configs and launch** (local):

```bash
# Preview what will be generated (dry run)
python script_utils/generate_inference_configs.py \
    --config_name search_motif_local_pipeline \
    --sweeper configs/sweeps/motif_tasks.yaml \
    --run_name motif_sweep \
    --dryrun

# Generate for real
python script_utils/generate_inference_configs.py \
    --config_name search_motif_local_pipeline \
    --sweeper configs/sweeps/motif_tasks.yaml \
    --run_name motif_sweep
```


See `docs/SWEEP.md` for full sweep documentation.

The `slurm_utils/launch_protein_binder_search_from_local_conda.sh` does all of this for you for the launcher and it can be replicated for motifs.
