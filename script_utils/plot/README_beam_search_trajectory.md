# Beam Search Trajectory: Saving and Visualization

This document explains how to save beam search trajectories during binder generation and how to visualize them interactively.

## Overview

Beam search expands candidates at each denoising checkpoint: from `beam_width` samples, it branches into `beam_width × n_branch` candidates, evaluates rewards via look-ahead completion, and keeps the top `beam_width`. Saving the trajectory records these intermediate states, look-ahead samples, and selection metadata so you can inspect the search tree.

## Saving Trajectories

### 1. Enable trajectory saving in config

Set `trajectory_dir` in the beam search section of your generation config (e.g. `configs/pipeline/binder_generate.yaml`):

```yaml
search:
  algorithm: beam-search
  step_checkpoints: [0, 100, 200, 300, 400]

  beam_search:
    n_branch: 2
    beam_width: 2
    trajectory_dir: ./beam_trajectory   # Required: path where trajectory is saved
    include_intermediate_denoising_states: true  # Optional: also save intermediate decoded states
```

- **`trajectory_dir`**: Path (relative or absolute) where the trajectory will be written. When set, beam search will create this directory and write all structures and metadata there.
- **`include_intermediate_denoising_states`**: If `true`, saves the decoded structure at each checkpoint *before* look-ahead roll-out. This shows the exact state that was selected and carried forward (useful for understanding what gets kept vs discarded).

### 2. Run generation

Run the binder design pipeline as usual:

```bash
complexa design configs/search_binder_local_pipeline.yaml ++generation.task_name=33_TrkA
```

Or override the trajectory path:

```bash
complexa design configs/search_binder_local_pipeline.yaml \
  ++generation.search.beam_search.trajectory_dir=/path/to/my_trajectory
```

### 3. Output structure

After a run with `trajectory_dir` set, you will see:

```
beam_trajectory/
├── trajectory_manifest.csv      # Index of all samples and metadata
├── step_0/                      # Look-ahead samples at step 0 (reward evaluated)
│   └── sample_0.pdb, sample_1.pdb, ...
├── step_0_intermediate/         # Intermediate decoded states at step 0 (if include_intermediate_denoising_states)
│   └── sample_0.pdb, sample_1.pdb, ...
├── step_1/
├── step_1_intermediate/
├── ...
├── step_N_final/                # Final outputs after last selection
│   └── sample_0.pdb, ...
```

- **`step_k/`**: Look-ahead samples—structures rolled out from the intermediate state at checkpoint `k` to completion, used for reward evaluation.
- **`step_k_intermediate/`**: Intermediate states—decoded latent at checkpoint `k`, before look-ahead. Selection keeps the *intermediate* state corresponding to the top look-ahead, not the look-ahead itself.
- **`step_N_final/`**: Final outputs (one per original sample × beam_width).

---

## Manifest Metadata

`trajectory_manifest.csv` is a CSV with one row per saved sample. Columns:

| Column | Description | Example |
|--------|-------------|---------|
| `step` | Search step index (0-based, corresponds to checkpoint index) | `0`, `1`, `2` |
| `end_step` | Denoising step at end of this segment (noise schedule) | `100`, `200`, `400` |
| `sample_id` | Unique ID for this sample within the manifest | `0`, `1`, `7` |
| `branch` | Branch index (`0..n_branch-1`) for this sample | `0`, `1` |
| `beam` | Beam index (`0..beam_width-1`) | `0`, `1` |
| `orig_sample` | Original sample index in the batch (`0..nsamples-1`) | `0`, `1` |
| `total_reward` | Reward score (look-ahead only; empty for intermediate/final) | `-0.598702` |
| `selected` | `1` if this sample was selected for the next step, `0` otherwise | `0`, `1` |
| `sample_type` | `intermediate`, `lookahead`, or `final` | `lookahead` |
| `parent_sample_id` | ID of the look-ahead sample from the *previous* step that led to this one. `-1` for step 0. | `-1`, `7` |
| `origin_intermediate_sample_id` | For look-ahead only: `sample_id` of the intermediate state this look-ahead originated from | `0`, `4` |
| `kept` | `1` if this sample is kept after selection, `0` if discarded | `0`, `1` |
| `pdb_path` | Path to PDB file, relative to trajectory directory | `step_0/sample_0.pdb` |

### Sample types

1. **`intermediate`**: Decoded structure at a checkpoint (flow-matching state). Selection is based on look-ahead rewards, but the *intermediate* state is what is carried forward to the next step.
2. **`lookahead`**: Roll-out from intermediate to full completion, used only for reward evaluation. Has `total_reward` and `origin_intermediate_sample_id` pointing to the originating intermediate.
3. **`final`**: Output after the last selection step. No look-ahead; all are considered kept.

### Parent / origin relationships

- **`parent_sample_id`**: Points to the look-ahead sample (from the previous step) whose reward caused this sample to be chosen. The next-step intermediate decodes from the latent corresponding to that parent’s *intermediate* (via `origin_intermediate_sample_id`).
- **`origin_intermediate_sample_id`**: For look-aheads only. Indicates which intermediate (same step) this look-ahead was simulated from.

---

## Visualization

### Basic usage

```bash
python script_utils/plot/visualize_beam_search_trajectory.py \
  --trajectory_dir path/to/beam_trajectory \
  --output trajectory_viz.html
```

Then open `trajectory_viz.html` in a browser. The page uses 3Dmol.js and requires internet access to load the library.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--trajectory_dir` | — | Directory with `trajectory_manifest.csv` and PDB subdirs |
| `--output` | `beam_search_trajectory.html` | Output HTML file path |
| `--max_samples_per_step` | `4` | Max samples per step in overlay view |
| `--show_reward_chart` | `True` | Include reward range summary per step |
| `--dryrun` | `False` | Show what would be done without writing files |

### Demo mode (no real trajectory)

To try the viewer without running beam search:

```bash
python script_utils/plot/visualize_beam_search_trajectory.py --demo --output demo_trajectory.html
```

Or from a custom PDB:

```bash
python script_utils/plot/visualize_beam_search_trajectory.py --demo --pdb path/to/protein.pdb --output demo_trajectory.html
```

Demo mode creates a synthetic trajectory with small perturbations at each step.

### What the visualization shows

- **Search tree**: Hierarchical view of intermediate → look-ahead → next intermediate. Kept nodes (green border) vs discarded (red).
- **3D structure viewer**: Click a node to view its PDB. Toggle "Show look-ahead samples" to include look-aheads in the tree.
- **Overlay**: "Overlay all steps" to superimpose one structure per step (different colors).
- **Orig sample filter**: Filter the tree by original batch sample.

---

## Example workflow

```bash
# 1. Run beam search with trajectory saving
complexa design configs/search_binder_local_pipeline.yaml \
  ++generation.task_name=33_TrkA \
  ++generation.search.beam_search.trajectory_dir=./my_run/beam_trajectory

# 2. Visualize
python script_utils/plot/visualize_beam_search_trajectory.py \
  --trajectory_dir ./my_run/beam_trajectory \
  --output my_run/beam_viz.html

# 3. Open in browser
xdg-open my_run/beam_viz.html   # or: open, firefox, etc.
```
