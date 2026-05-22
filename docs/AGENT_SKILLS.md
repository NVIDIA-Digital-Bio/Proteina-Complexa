# Agent Skills

Proteina-Complexa ships a set of project-local [Claude Code Agent Skills](https://docs.claude.com/en/docs/claude-code/skills) under [`.claude/skills/`](../.claude/skills/). They give Claude Code (and any other Anthropic-Skill-aware agent) detailed, opinionated guidance on how to drive the `complexa` CLI for the most common workflows in this repo — from a fresh setup to a finished design run with a replayable manifest.

> **Documentation Map**
> - Running a design end-to-end? See [Inference Guide](INFERENCE.md)
> - Tuning YAML configs? See [Configuration Guide](CONFIGURATION_GUIDE.md)
> - Understanding the evaluate / analyze metrics? See [Evaluation Guide](EVALUATION_METRICS.md)
> - Parameter sweeps? See [Sweep System](SWEEP.md) and the `complexa-sweep` skill below

## What an Agent Skill is

Each skill is a directory containing a `SKILL.md` with YAML front-matter (`name`, `description`, `compatibility`, `allowed-tools`) plus optional progressive-disclosure references under `reference/` and helper scripts under `scripts/`. Anthropic's spec is documented [here](https://github.com/anthropics/skills).

When Claude Code is invoked inside this repo (`cd Proteina-Complexa && claude`), it scans `.claude/skills/*/SKILL.md`, indexes the descriptions, and loads the matching skill body verbatim into context the moment the user's prompt looks like the skill is needed (e.g. "design a binder for PDL1" pulls in `complexa-design`). No setup or registration step is required on your end — the skills travel with the repo.

## Skill catalog

All five skills live at [`.claude/skills/`](../.claude/skills/). The README in that directory has the full "pipeline cheat sheet" and "primary tool per skill" table; the catalog below is the one-line index.

| Skill | What it drives | When to invoke |
|---|---|---|
| [`complexa-setup`](../.claude/skills/complexa-setup/SKILL.md) | `complexa init`, `complexa download`, `complexa validate env` | Fresh clone, configuring `.env`, downloading model weights |
| [`complexa-target`](../.claude/skills/complexa-target/SKILL.md) | `complexa target add/list/show/validate` + direct edits to `configs/targets/{,ligand_}targets_dict.yaml` | Registering a new protein or ligand design target |
| [`complexa-design`](../.claude/skills/complexa-design/SKILL.md) | `complexa design <pipeline>` for all three pipelines (protein binder, ligand binder, AME) | End-to-end design run: generate → filter → evaluate → analyze |
| [`complexa-evaluate-pdbs`](../.claude/skills/complexa-evaluate-pdbs/SKILL.md) | `complexa analysis configs/evaluate_*_from_pdb_dir.yaml ++sample_storage_path=<dir>` | Score an existing PDB directory with AF2 / RF3 / ESMFold (handy for third-party designs from BindCraft, RFdiffusion, etc.) |
| [`complexa-sweep`](../.claude/skills/complexa-sweep/SKILL.md) | [`script_utils/generate_inference_configs.py`](../script_utils/generate_inference_configs.py) + a `complexa design` loop over the generated configs | Cartesian-product hyperparameter sweeps (beam width, nsteps, reward weights, …) |

Shared helpers live under [`.claude/skills/_shared/`](../.claude/skills/_shared/):

| File | Purpose |
|---|---|
| [`scripts/preflight.sh`](../.claude/skills/_shared/scripts/preflight.sh) | One-shot host probe (GPU, VRAM, disk, ckpts, tool binaries, `.env`). Emits `preflight.json` that the design / evaluate / sweep skills read in Step 1. |
| [`scripts/write_manifest.py`](../.claude/skills/_shared/scripts/write_manifest.py) | Writes a pinned, replayable `run_manifest.json` per pipeline run (command, git SHA, ckpt SHA-256s, CSV pointers). |
| [`reference/hardware.md`](../.claude/skills/_shared/reference/hardware.md) | Per-pipeline VRAM / CPU / disk requirements and a "when you hit OOM" mitigation checklist. |

## Using them

If you're running the [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) from this repo, you're already set — there is nothing to install. To verify that Claude Code sees the skills, ask it `list available skills` from a session inside the repo; the five `complexa-*` names should appear.

For a quick "is this working" sanity check from inside a Claude Code session:

```text
> Show me the protein binder design pipeline overview from the complexa-design skill.
```

Claude should respond by loading the skill body and summarising it (instead of grepping configs from scratch).

If you are not using Claude Code, the skill bodies are still useful as standalone documentation — they cite specific source files in this repo and give worked examples for every common invocation. Read them top-down in this order: `complexa-setup` → `complexa-target` → `complexa-design` → `complexa-evaluate-pdbs` → `complexa-sweep`.

## Authoring new skills

These were built with Anthropic's [`skill-creator`](https://github.com/anthropics/skills/tree/main/skill-creator) workflow:

1. **Draft** a `SKILL.md` (≤300 lines, YAML front-matter + Markdown body) plus optional `reference/*.md` files for progressive disclosure. Anchor every claim to a specific source file in this repo.
2. **Write test prompts** that should trigger the skill, then evaluate with-skill vs without-skill against objective assertions (uses the right flag, cites a real config key, runs preflight, etc.).
3. **Iterate** on regressions.

To add a new skill: drop a `your-skill/SKILL.md` into `.claude/skills/`, include a `description:` keyword-rich enough that Claude Code will route to it for the prompts you care about, and consider following the same Step 1 (preflight) / Step N (manifest) bookends the existing skills use.