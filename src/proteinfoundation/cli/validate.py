"""
Validation utilities for Complexa CLI.

Validates configurations and checks for required files/models before execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class ValidationResult:
    """Result of a validation check."""

    name: str
    passed: bool
    message: str
    fix_hint: str | None = None
    is_warning: bool = False  # Warning vs error


@dataclass
class ValidationReport:
    """Collection of validation results."""

    results: list[ValidationResult] = field(default_factory=list)
    validation_type: str = ""
    config_path: Path | None = None

    def add(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)

    def add_pass(self, name: str, message: str) -> None:
        """Add a passing result."""
        self.results.append(ValidationResult(name=name, passed=True, message=message))

    def add_fail(
        self,
        name: str,
        message: str,
        fix_hint: str | None = None,
        is_warning: bool = False,
    ) -> None:
        """Add a failing result."""
        self.results.append(
            ValidationResult(
                name=name,
                passed=False,
                message=message,
                fix_hint=fix_hint,
                is_warning=is_warning,
            )
        )

    @property
    def all_passed(self) -> bool:
        """Check if all validations passed (ignoring warnings)."""
        return all(r.passed or r.is_warning for r in self.results)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors (not warnings)."""
        return any(not r.passed and not r.is_warning for r in self.results)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(not r.passed and r.is_warning for r in self.results)

    def print_report(self) -> None:
        """Print the validation report."""
        # ANSI colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        RESET = "\033[0m"

        print(f"\n{'=' * 70}")
        print(f"  {BOLD}🔍 Validation Report: {self.validation_type}{RESET}")
        if self.config_path:
            print(f"  {DIM}Config: {self.config_path}{RESET}")
        print(f"{'=' * 70}\n")

        # Group by category (passed/failed)
        passed = [r for r in self.results if r.passed]
        warnings = [r for r in self.results if not r.passed and r.is_warning]
        errors = [r for r in self.results if not r.passed and not r.is_warning]

        if errors:
            print(f"  {RED}{BOLD}✗ Errors ({len(errors)}):{RESET}")
            for r in errors:
                print(f"    {RED}✗{RESET} {r.name}")
                print(f"      {DIM}{r.message}{RESET}")
                if r.fix_hint:
                    print(f"      {CYAN}💡 {r.fix_hint}{RESET}")
            print()

        if warnings:
            print(f"  {YELLOW}{BOLD}⚠ Warnings ({len(warnings)}):{RESET}")
            for r in warnings:
                print(f"    {YELLOW}⚠{RESET} {r.name}")
                print(f"      {DIM}{r.message}{RESET}")
                if r.fix_hint:
                    print(f"      {CYAN}💡 {r.fix_hint}{RESET}")
            print()

        if passed:
            print(f"  {GREEN}{BOLD}✓ Passed ({len(passed)}):{RESET}")
            for r in passed:
                print(f"    {GREEN}✓{RESET} {r.name}: {r.message}")
            print()

        # Summary
        print(f"{'─' * 70}")
        if self.all_passed:
            if self.has_warnings:
                print(f"  {YELLOW}⚠ Validation passed with warnings{RESET}")
            else:
                print(f"  {GREEN}✓ All validations passed!{RESET}")
        else:
            print(f"  {RED}✗ Validation failed - please fix errors before running{RESET}")
        print(f"{'=' * 70}\n")


def load_env_config() -> dict[str, str]:
    """Load environment configuration from .env file."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)

    # Return relevant env vars
    env_vars = [
        "DATA_PATH",
        "AF2_DIR",
        "RF3_DIR",
        "RF3_CKPT_PATH",
        "RF3_EXEC_PATH",
        "FOLDSEEK_EXEC",
        "SC_EXEC",
        "MMSEQS_EXEC",
        "DSSP_EXEC",
    ]

    return {key: os.environ.get(key, "") for key in env_vars}


def _resolve_oc_env(value):
    """Resolve ${oc.env:VAR} and ${oc.env:VAR,default} patterns from environment."""
    if not isinstance(value, str) or "${" not in value:
        return value
    import re

    def _replace(m):
        var = m.group(1)
        default = m.group(2)
        env_val = os.environ.get(var)
        if env_val is not None:
            return env_val
        return default if default is not None else m.group(0)

    return re.sub(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}", _replace, value)


def _resolve_config(obj):
    """Recursively resolve ${oc.env:...} in a config dict."""
    if isinstance(obj, dict):
        return {k: _resolve_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_config(v) for v in obj]
    return _resolve_oc_env(obj)


def load_config_yaml(config_path: Path) -> dict:
    """Load and return a YAML config file, resolving ${oc.env:...} from environment."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return _resolve_config(raw)


def check_path_exists(path: str | Path, name: str) -> ValidationResult:
    """Check if a path exists."""
    path = Path(path) if path else None
    if not path:
        return ValidationResult(
            name=name,
            passed=False,
            message="Path not configured",
            fix_hint="Run 'complexa init' to set up environment",
        )
    if path.exists():
        return ValidationResult(name=name, passed=True, message=str(path))
    return ValidationResult(
        name=name,
        passed=False,
        message=f"Path not found: {path}",
        fix_hint="Check that the path exists or run 'complexa download'",
    )


def check_file_exists(path: str | Path, name: str) -> ValidationResult:
    """Check if a file exists."""
    path = Path(path) if path else None
    if not path:
        return ValidationResult(
            name=name,
            passed=False,
            message="File path not configured",
            fix_hint="Check your configuration",
        )
    if path.is_file():
        return ValidationResult(name=name, passed=True, message=str(path))
    return ValidationResult(
        name=name,
        passed=False,
        message=f"File not found: {path}",
        fix_hint="Ensure the file exists at the specified path",
    )


def check_dir_exists(path: str | Path, name: str) -> ValidationResult:
    """Check if a directory exists."""
    path = Path(path) if path else None
    if not path:
        return ValidationResult(
            name=name,
            passed=False,
            message="Directory path not configured",
            fix_hint="Check your configuration",
        )
    if path.is_dir():
        return ValidationResult(name=name, passed=True, message=str(path))
    return ValidationResult(
        name=name,
        passed=False,
        message=f"Directory not found: {path}",
        fix_hint="Create the directory or check the path",
    )


def validate_env() -> ValidationReport:
    """Validate the .env file exists and has required variables."""
    report = ValidationReport(validation_type="Environment")

    env_path = Path(".env")
    if not env_path.exists():
        report.add_fail(
            ".env file",
            "No .env file found in current directory",
            fix_hint="Run 'complexa init' to create one",
        )
        return report

    report.add_pass(".env file", "Found")

    # Load env
    env = load_env_config()

    # Check critical paths
    if env.get("DATA_PATH"):
        report.add(check_dir_exists(env["DATA_PATH"], "DATA_PATH"))
    else:
        report.add_fail("DATA_PATH", "Not set in .env", fix_hint="Run 'complexa init'")

    return report


def _get_target_dict_cfg(config_path: Path) -> tuple[dict | None, str | None]:
    """Extract target_dict_cfg and task_name from config.

    Follows Hydra defaults chain to find target_dict_cfg:
    1. Check main config for target_dict_cfg
    2. Follow generation defaults (e.g., targets_search_local.yaml)
    3. Follow nested defaults (e.g., targets_dict.yaml)

    Returns
    -------
    tuple
        (target_dict_cfg, task_name) - either can be None if not found
    """
    try:
        cfg = load_config_yaml(config_path)
    except Exception:
        return None, None

    target_dict_cfg = None
    task_name = None
    generation_dir = config_path.parent / "generation"

    # Get task_name from top-level generation config (highest priority)
    if "generation" in cfg:
        gen_cfg = cfg["generation"]
        task_name = gen_cfg.get("task_name")
        target_dict_cfg = gen_cfg.get("target_dict_cfg")

    # Follow defaults chain to find generation config and targets_dict
    if "defaults" in cfg:
        for default in cfg["defaults"]:
            if isinstance(default, dict) and "generation" in default:
                gen_file = generation_dir / f"{default['generation']}.yaml"
                if gen_file.exists():
                    gen_cfg = load_config_yaml(gen_file)

                    # Get task_name from generation config if not already set
                    if not task_name:
                        # Check dataloader.dataset.task_name
                        dataloader = gen_cfg.get("dataloader", {})
                        dataset = dataloader.get("dataset", {})
                        task_name = dataset.get("task_name")

                    # Check for target_dict_cfg in generation config
                    if not target_dict_cfg:
                        target_dict_cfg = gen_cfg.get("target_dict_cfg")

                    # Follow nested defaults in generation config (e.g., targets_dict)
                    if not target_dict_cfg and "defaults" in gen_cfg:
                        for nested_default in gen_cfg["defaults"]:
                            # Handle string defaults like "- targets_dict"
                            if isinstance(nested_default, str):
                                nested_file = generation_dir / f"{nested_default}.yaml"
                                if nested_file.exists():
                                    nested_cfg = load_config_yaml(nested_file)
                                    # targets_dict.yaml has target_dict_cfg at top level
                                    if "target_dict_cfg" in nested_cfg:
                                        target_dict_cfg = nested_cfg["target_dict_cfg"]
                                        break
                                    # Or the whole file might be the dict
                                    elif nested_default == "targets_dict" and nested_cfg:
                                        # Check if it looks like a targets dict (has target entries)
                                        first_key = next(iter(nested_cfg.keys()), None)
                                        if first_key and isinstance(nested_cfg.get(first_key), dict):
                                            if (
                                                "source" in nested_cfg[first_key]
                                                or "target_input" in nested_cfg[first_key]
                                            ):
                                                target_dict_cfg = nested_cfg
                                                break
                    break

    # Fallback: try loading targets_dict.yaml directly
    if not target_dict_cfg:
        targets_dict_path = generation_dir / "targets_dict.yaml"
        if targets_dict_path.exists():
            try:
                targets_cfg = load_config_yaml(targets_dict_path)
                # Check if it has target_dict_cfg key or is the dict itself
                if "target_dict_cfg" in targets_cfg:
                    target_dict_cfg = targets_cfg["target_dict_cfg"]
                elif targets_cfg:
                    # Check if it looks like a targets dict
                    first_key = next(iter(targets_cfg.keys()), None)
                    if first_key and isinstance(targets_cfg.get(first_key), dict):
                        if "source" in targets_cfg[first_key] or "target_input" in targets_cfg[first_key]:
                            target_dict_cfg = targets_cfg
            except Exception:
                pass

    return target_dict_cfg, task_name


def validate_target(
    config_path: Path | None = None,
    target_name: str | None = None,
) -> ValidationReport:
    """Validate target configuration and file accessibility."""
    report = ValidationReport(validation_type="Target", config_path=config_path)

    # Load environment
    env = load_env_config()
    data_path = env.get("DATA_PATH", "")

    if not data_path:
        report.add_fail(
            "DATA_PATH",
            "DATA_PATH not set",
            fix_hint="Run 'complexa init' to configure environment",
        )
        return report

    report.add(check_dir_exists(data_path, "DATA_PATH"))

    # Check target_data directory exists
    target_data_dir = Path(data_path) / "target_data"
    report.add(check_dir_exists(target_data_dir, "target_data directory"))

    # If config provided, extract target info
    if config_path and config_path.exists():
        try:
            target_dict_cfg, config_task_name = _get_target_dict_cfg(config_path)

            # Use provided target_name or fall back to config's task_name
            effective_target = target_name or config_task_name

            if effective_target and str(effective_target).startswith("${"):
                report.add_fail(
                    "Target name",
                    f"Target name is an unresolved interpolation: {effective_target}",
                    fix_hint="Set generation.task_name explicitly in the config",
                    is_warning=True,
                )
                return report

            if effective_target:
                report.add_pass("Target name", str(effective_target))

                if target_dict_cfg:
                    if effective_target in target_dict_cfg:
                        target_cfg = target_dict_cfg[effective_target]

                        # Build target path
                        if target_cfg.get("target_path"):
                            target_path = Path(target_cfg["target_path"])
                            path_source = "target_path (explicit)"
                        else:
                            source = target_cfg.get("source", "")
                            filename = target_cfg.get("target_filename", "")
                            target_path = Path(data_path) / "target_data" / source / f"{filename}.pdb"
                            path_source = f"DATA_PATH/target_data/{source}/{filename}.pdb"

                        # Report the path being checked
                        report.add_pass("Target path source", path_source)

                        # Check if file exists
                        result = check_file_exists(target_path, "Target PDB file")
                        if result.passed:
                            result.message = str(target_path)
                        report.add(result)

                        # Report additional target info
                        target_input = target_cfg.get("target_input", "")
                        if target_input:
                            report.add_pass("Target input spec", str(target_input))

                        hotspots = target_cfg.get("hotspot_residues", [])
                        if hotspots:
                            report.add_pass(
                                "Hotspot residues",
                                ", ".join(str(h) for h in hotspots if h),
                            )

                        binder_len = target_cfg.get("binder_length", [])
                        if binder_len:
                            if isinstance(binder_len, list) and len(binder_len) == 2:
                                report.add_pass("Binder length", f"{binder_len[0]}-{binder_len[1]}")
                            else:
                                report.add_pass("Binder length", str(binder_len))

                        if "ligand" in target_cfg:
                            report.add_pass("Target type", "Ligand")
                        else:
                            report.add_pass("Target type", "Protein")

                    else:
                        report.add_fail(
                            f"Target '{effective_target}'",
                            f"Target not found in target_dict_cfg ({len(target_dict_cfg)} targets available)",
                            fix_hint="Check target name or run 'complexa target list'",
                        )
                else:
                    report.add_fail(
                        "Target config",
                        "Could not find target_dict_cfg in config",
                        fix_hint="Check config defaults or targets_dict.yaml",
                    )
            else:
                report.add_fail(
                    "Target name",
                    "No task_name found in config and no --target provided",
                    fix_hint="Set generation.task_name in config or use --target NAME",
                )

        except Exception as e:
            report.add_fail("Config parsing", str(e))

    return report


def validate_generate(config_path: Path) -> ValidationReport:
    """Validate configuration for the generate step."""
    report = ValidationReport(validation_type="Generate", config_path=config_path)

    # Check config exists
    if not config_path.exists():
        report.add_fail("Config file", f"Not found: {config_path}")
        return report
    report.add_pass("Config file", str(config_path))

    # Load environment
    env = load_env_config()

    # Check DATA_PATH
    if env.get("DATA_PATH"):
        report.add(check_dir_exists(env["DATA_PATH"], "DATA_PATH"))
    else:
        report.add_fail("DATA_PATH", "Not set", fix_hint="Run 'complexa init'")

    # Load config
    try:
        cfg = load_config_yaml(config_path)
    except Exception as e:
        report.add_fail("Config parsing", str(e))
        return report

    # Check for Complexa model checkpoint
    # ckpt_path can be a directory, ckpt_name is the filename
    ckpt_path = cfg.get("ckpt_path")
    ckpt_name = cfg.get("ckpt_name")

    if ckpt_path and "${" not in str(ckpt_path):
        ckpt_path = Path(ckpt_path)

        # If ckpt_name is provided, combine them
        if ckpt_name and "${" not in str(ckpt_name):
            full_ckpt_path = ckpt_path / ckpt_name
            report.add(check_file_exists(full_ckpt_path, "Complexa checkpoint"))
        elif ckpt_path.is_file():
            # ckpt_path is already a full file path
            report.add(check_file_exists(ckpt_path, "Complexa checkpoint"))
        elif ckpt_path.is_dir():
            # ckpt_path is a directory but no ckpt_name - check for .ckpt files
            ckpt_files = list(ckpt_path.glob("*.ckpt"))
            if ckpt_files:
                report.add_pass(
                    "Complexa checkpoint directory",
                    f"{ckpt_path} ({len(ckpt_files)} .ckpt files)",
                )
            else:
                report.add_fail(
                    "Complexa checkpoint",
                    f"Directory exists but no .ckpt files found: {ckpt_path}",
                    fix_hint="Add ckpt_name to config or place .ckpt files in directory",
                )
        else:
            report.add_fail(
                "Complexa checkpoint",
                f"Path not found: {ckpt_path}",
                fix_hint="Check ckpt_path in config",
            )
    else:
        report.add_fail(
            "Complexa checkpoint",
            "ckpt_path not found in config",
            fix_hint="Add ckpt_path to your config or check defaults",
            is_warning=True,
        )

    # Check autoencoder checkpoint if specified
    ae_ckpt_path = cfg.get("autoencoder_ckpt_path")
    if ae_ckpt_path and "${" not in str(ae_ckpt_path):
        report.add(check_file_exists(ae_ckpt_path, "Autoencoder checkpoint"))

    # Check target
    target_name = None
    if "generation" in cfg:
        gen_cfg = cfg["generation"]
        target_name = gen_cfg.get("task_name")

    if target_name and not str(target_name).startswith("${"):
        target_report = validate_target(config_path, target_name)
        for r in target_report.results:
            if "Target PDB" in r.name or "target_data" in r.name:
                report.add(r)

    # Check search/reward configuration
    gen_cfg = cfg.get("generation", {})
    search_cfg = gen_cfg.get("search", {})
    if search_cfg:
        algorithm = search_cfg.get("algorithm", "best-of-n")
        if algorithm != "best-of-n":
            report.add_pass("Search algorithm", algorithm)

    # Check reward model if enabled
    reward_cfg = gen_cfg.get("reward_model", {})
    if reward_cfg:
        reward_target = reward_cfg.get("_target_", "")
        # For CompositeRewardModel, also check nested reward_models
        reward_targets = [reward_target]
        for nested in reward_cfg.get("reward_models", {}).values():
            if hasattr(nested, "get"):
                reward_targets.append(nested.get("_target_", ""))

        if any("AF2" in rt or "alphafold" in rt.lower() for rt in reward_targets):
            af2_dir = env.get("AF2_DIR", "")
            if af2_dir:
                report.add(check_dir_exists(af2_dir, "AF2 weights (for reward)"))
            else:
                report.add_fail(
                    "AF2 weights (for reward)",
                    "AF2_DIR not set",
                    fix_hint="Run 'complexa init' or 'complexa download'",
                )

        if any("RF3" in rt or "rosetta" in rt.lower() for rt in reward_targets):
            rf3_ckpt = env.get("RF3_CKPT_PATH", "")
            if rf3_ckpt:
                report.add(check_file_exists(rf3_ckpt, "RF3 checkpoint (for reward)"))
            else:
                report.add_fail(
                    "RF3 checkpoint (for reward)",
                    "RF3_CKPT_PATH not set",
                    fix_hint="Run 'complexa init' or download RF3 weights",
                )

    return report


def validate_evaluate(config_path: Path) -> ValidationReport:
    """Validate configuration for the evaluate step."""
    report = ValidationReport(validation_type="Evaluate", config_path=config_path)

    # Check config exists
    if not config_path.exists():
        report.add_fail("Config file", f"Not found: {config_path}")
        return report
    report.add_pass("Config file", str(config_path))

    # Load environment
    env = load_env_config()

    # Check DATA_PATH
    if env.get("DATA_PATH"):
        report.add(check_dir_exists(env["DATA_PATH"], "DATA_PATH"))
    else:
        report.add_fail("DATA_PATH", "Not set", fix_hint="Run 'complexa init'")

    # Load config
    try:
        cfg = load_config_yaml(config_path)
    except Exception as e:
        report.add_fail("Config parsing", str(e))
        return report

    # Check evaluation config
    eval_cfg = cfg.get("evaluation", {})
    if not eval_cfg:
        # Try to find eval defaults
        if "defaults" in cfg:
            for default in cfg["defaults"]:
                if isinstance(default, dict) and "evaluation" in default:
                    eval_file = config_path.parent / "evaluation" / f"{default['evaluation']}.yaml"
                    if eval_file.exists():
                        eval_cfg = load_config_yaml(eval_file)
                        break

    # Check folding models
    folding_models = eval_cfg.get("folding_models", [])
    if not folding_models:
        # Check metric config for folding models - can be at top level or under generation
        metric_cfg = cfg.get("metric", {})
        if not metric_cfg:
            metric_cfg = cfg.get("generation", {}).get("metric", {})

        if metric_cfg.get("compute_binder_metrics"):
            folding_method = metric_cfg.get("binder_folding_method", "colabdesign")
            folding_models = [folding_method]
            report.add_pass("Binder folding method", folding_method)

    for model in folding_models:
        model_lower = model.lower() if isinstance(model, str) else ""

        if "af2" in model_lower or "alphafold" in model_lower or "colabdesign" in model_lower:
            af2_dir = env.get("AF2_DIR", "")
            if af2_dir:
                # Check for model weights in AF2 directory
                af2_path = Path(af2_dir)
                if af2_path.exists():
                    # Check for params subdirectory, .npz files, or .pkl files
                    npz_files = list(af2_path.glob("*.npz"))
                    pkl_files = list(af2_path.glob("*.pkl"))
                    has_params_dir = (af2_path / "params").exists()

                    if npz_files:
                        report.add_pass("AF2 weights", f"{af2_path} ({len(npz_files)} .npz files)")
                    elif pkl_files:
                        report.add_pass("AF2 weights", f"{af2_path} ({len(pkl_files)} .pkl files)")
                    elif has_params_dir:
                        report.add_pass("AF2 weights", f"{af2_path}/params")
                    else:
                        report.add_fail(
                            "AF2 weights",
                            f"Directory exists but no weights found: {af2_path}",
                            fix_hint="Run 'complexa download' to download AF2 weights",
                        )
                else:
                    report.add_fail(
                        "AF2 weights",
                        f"Directory not found: {af2_path}",
                        fix_hint="Run 'complexa download' to download AF2 weights",
                    )
            else:
                report.add_fail(
                    "AF2 weights",
                    "AF2_DIR not set",
                    fix_hint="Run 'complexa init' and 'complexa download'",
                )

        if "rf3" in model_lower or "rosetta" in model_lower:
            rf3_ckpt = env.get("RF3_CKPT_PATH", "")
            if rf3_ckpt:
                report.add(check_file_exists(rf3_ckpt, "RF3 checkpoint"))
            else:
                report.add_fail(
                    "RF3 checkpoint",
                    "RF3_CKPT_PATH not set",
                    fix_hint="Run 'complexa init' and download RF3 weights",
                )

            rf3_exec = env.get("RF3_EXEC_PATH", "")
            if rf3_exec:
                report.add(check_path_exists(rf3_exec, "RF3 executable"))
            else:
                report.add_fail(
                    "RF3 executable",
                    "RF3_EXEC_PATH not set",
                    fix_hint="Run 'complexa init'",
                    is_warning=True,
                )

        if "boltz" in model_lower:
            # Boltz downloads weights automatically, just warn
            report.add_pass("Boltz2", "Will download weights if needed")

        if "esmfold" in model_lower:
            # ESMFold downloads weights automatically
            report.add_pass("ESMFold", "Will download weights if needed")

    # Check for external tools used in evaluation
    tools = [
        ("FOLDSEEK_EXEC", "Foldseek"),
        ("SC_EXEC", "Shape complementarity (sc)"),
    ]

    for env_key, tool_name in tools:
        tool_path = env.get(env_key, "")
        if tool_path:
            result = check_file_exists(tool_path, tool_name)
            result.is_warning = True  # Tools are optional
            report.add(result)

    return report


def validate_analyze(config_path: Path) -> ValidationReport:
    """Validate configuration for the analyze step."""
    report = ValidationReport(validation_type="Analyze", config_path=config_path)

    # Check config exists
    if not config_path.exists():
        report.add_fail("Config file", f"Not found: {config_path}")
        return report
    report.add_pass("Config file", str(config_path))

    # Check for evaluation results directory
    eval_dir = Path("./evaluation_results")
    if eval_dir.exists():
        report.add_pass("Evaluation results directory", str(eval_dir))
    else:
        report.add_fail(
            "Evaluation results directory",
            "Not found - run evaluate step first",
            fix_hint="Run 'complexa evaluate' before 'complexa analyze'",
            is_warning=True,
        )

    return report


def validate_design(config_path: Path) -> ValidationReport:
    """Validate configuration for the full design pipeline."""
    report = ValidationReport(validation_type="Design Pipeline", config_path=config_path)

    # Run all validations
    env_report = validate_env()
    gen_report = validate_generate(config_path)
    eval_report = validate_evaluate(config_path)

    # Merge results (avoid duplicates)
    seen_names = set()

    for r in env_report.results:
        if r.name not in seen_names:
            report.add(r)
            seen_names.add(r.name)

    for r in gen_report.results:
        if r.name not in seen_names:
            report.add(r)
            seen_names.add(r.name)

    for r in eval_report.results:
        if r.name not in seen_names:
            report.add(r)
            seen_names.add(r.name)

    return report


def run_validation(
    validation_type: str,
    config_path: Path | None = None,
    target_name: str | None = None,
) -> ValidationReport:
    """Run validation based on type.

    Parameters
    ----------
    validation_type : str
        One of: 'target', 'generate', 'evaluate', 'analyze', 'design', 'env'
    config_path : Path, optional
        Path to config file (required for most validations)
    target_name : str, optional
        Target name to validate (for target validation)

    Returns
    -------
    ValidationReport
        The validation report
    """
    if validation_type == "env":
        return validate_env()
    elif validation_type == "target":
        return validate_target(config_path, target_name)
    elif validation_type == "generate":
        if not config_path:
            report = ValidationReport(validation_type="Generate")
            report.add_fail("Config", "Config path required for generate validation")
            return report
        return validate_generate(config_path)
    elif validation_type == "evaluate":
        if not config_path:
            report = ValidationReport(validation_type="Evaluate")
            report.add_fail("Config", "Config path required for evaluate validation")
            return report
        return validate_evaluate(config_path)
    elif validation_type == "analyze":
        if not config_path:
            report = ValidationReport(validation_type="Analyze")
            report.add_fail("Config", "Config path required for analyze validation")
            return report
        return validate_analyze(config_path)
    elif validation_type == "design":
        if not config_path:
            report = ValidationReport(validation_type="Design")
            report.add_fail("Config", "Config path required for design validation")
            return report
        return validate_design(config_path)
    else:
        report = ValidationReport(validation_type=validation_type)
        report.add_fail("Validation type", f"Unknown type: {validation_type}")
        return report
