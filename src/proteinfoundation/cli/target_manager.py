#!/usr/bin/env python3
"""
Target management utilities for Complexa CLI.

Provides functionality to list, add, and manage target configurations.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# =============================================================================
# Constants
# =============================================================================

# Default path to targets_dict.yaml (relative to project root)
# Primary location is configs/targets/, fallback to configs/generation/ for backwards compatibility
DEFAULT_TARGETS_DICT_PATH = Path("configs/targets/targets_dict.yaml")
FALLBACK_TARGETS_DICT_PATH = Path("configs/generation/targets_dict.yaml")

# Required fields for a target entry
# Note: Either (source AND target_filename) OR target_path is required, plus target_input
REQUIRED_FIELDS_ALWAYS = ["target_input"]
REQUIRED_FIELDS_PATH_OPTION = [
    "source",
    "target_filename",
]  # Required if target_path not provided

# All possible fields with their defaults and descriptions
TARGET_FIELDS = {
    "source": {
        "default": "custom_targets",
        "description": "Source directory under target_data/ (e.g., 'bindcraft_targets', 'alpha_proteo_targets')",
        "required": False,  # Required only if target_path not provided
    },
    "target_filename": {
        "default": None,
        "description": "Filename of the target PDB (without .pdb extension)",
        "required": False,  # Required only if target_path not provided
    },
    "target_path": {
        "default": None,
        "description": "Full path to target PDB file (if set, source/target_filename are optional)",
        "required": False,  # Required only if source/target_filename not provided
    },
    "target_input": {
        "default": "A1-100",
        "description": "Chain and residue range (e.g., 'A1-115' or 'A1-50,B1-50')",
        "required": True,
    },
    "hotspot_residues": {
        "default": [],
        "description": "List of hotspot residues (e.g., ['A33', 'A95'])",
        "required": False,
    },
    "binder_length": {
        "default": [60, 120],
        "description": "Binder length range [min, max] or single value [length]",
        "required": False,
    },
    "pdb_id": {
        "default": None,
        "description": "PDB ID for reference (optional)",
        "required": False,
    },
    "ligand": {
        "default": None,
        "description": "Ligand residue name(s) — str or list of str (presence implies ligand target)",
        "required": False,
    },
    # Ligand-specific fields (presence of 'ligand' key implies ligand target)
    # Note: The old "res_name" field has been renamed to "ligand" for consistency
    # with AME configs. The old "is_ligand" flag is no longer needed.
    "ligand_only": {
        "default": True,
        "description": "If True, generate binding pocket around ligand only (no protein-protein interface)",
        "required": False,
        "ligand_only": True,
    },
    "SMILES": {
        "default": None,
        "description": "SMILES string for the ligand molecule",
        "required": False,
        "ligand_only": True,
    },
    "use_bonds_from_file": {
        "default": True,
        "description": "If True, use bond information from the input PDB/CIF file",
        "required": False,
        "ligand_only": True,
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def find_project_root() -> Path:
    """Find the project root directory (contains configs/).

    Checks for configs/targets/ first (new location), then configs/generation/ (legacy).
    """
    current = Path.cwd()

    # Check current directory first (prefer configs/targets, fallback to configs/generation)
    if (current / "configs" / "targets").exists():
        return current
    if (current / "configs" / "generation").exists():
        return current

    # Check parent directories
    for parent in current.parents:
        if (parent / "configs" / "targets").exists():
            return parent
        if (parent / "configs" / "generation").exists():
            return parent

    # Fallback to current directory
    return current


def get_default_dict_path() -> Path:
    """Get the default path to targets_dict.yaml.

    Checks configs/targets/ first (new location), then configs/generation/ (legacy).
    """
    project_root = find_project_root()

    # Check primary location first
    primary_path = project_root / DEFAULT_TARGETS_DICT_PATH
    if primary_path.exists():
        return primary_path

    # Fallback to legacy location
    fallback_path = project_root / FALLBACK_TARGETS_DICT_PATH
    if fallback_path.exists():
        return fallback_path

    # Return primary path even if it doesn't exist (for error messages)
    return primary_path


def load_targets_dict(dict_path: Path | None = None) -> tuple[dict, Path]:
    """Load the targets dictionary from a YAML file.

    Parameters
    ----------
    dict_path : Path, optional
        Path to the targets dict file. If None, uses the default path.

    Returns
    -------
    tuple[dict, Path]
        The loaded dictionary and the resolved path
    """
    if dict_path is None:
        dict_path = get_default_dict_path()

    dict_path = Path(dict_path)

    if not dict_path.exists():
        raise FileNotFoundError(f"Targets dict not found: {dict_path}")

    with open(dict_path) as f:
        data = yaml.safe_load(f)

    return data, dict_path


class _FlowStyleListDumper(yaml.SafeDumper):
    """Custom YAML dumper that uses flow style for lists (inline [a, b, c] format)."""


def _represent_list(dumper, data):
    """Represent lists in flow style (inline) with quoted string items."""
    # Quote all string items in the list
    quoted_data = []
    for item in data:
        if isinstance(item, str):
            # Create a quoted string node
            quoted_data.append(item)
        else:
            quoted_data.append(item)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", quoted_data, flow_style=True)


def _represent_str(dumper, data):
    """Represent strings with quotes for consistency."""
    # Always quote strings that:
    # - Contain special characters
    # - Look like chain/residue patterns (e.g., A1-115, A33, B17-209)
    # - Are residue identifiers in hotspot_residues
    import re

    # Pattern for chain-residue ranges like A1-115, B17-209, or single residues like A33
    chain_residue_pattern = re.compile(r"^[A-Z]\d+(-\d+)?$")

    # Always quote if contains special chars or looks like a chain/residue pattern
    needs_quotes = (
        any(
            c in data
            for c in [
                ",",
                ":",
                "[",
                "]",
                "{",
                "}",
                "#",
                "&",
                "*",
                "!",
                "|",
                ">",
                "'",
                "%",
                "@",
                "`",
                "+",
            ]
        )
        or chain_residue_pattern.match(data)
        or data == "LIGAND"  # Special case for ligand targets
        or "-" in data  # Any string with a dash (ranges, SMILES, etc.)
    )

    if needs_quotes:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


# Register representers
_FlowStyleListDumper.add_representer(list, _represent_list)
_FlowStyleListDumper.add_representer(str, _represent_str)


def save_targets_dict(data: dict, dict_path: Path) -> None:
    """Save the targets dictionary to a YAML file (full rewrite).

    Parameters
    ----------
    data : dict
        The targets dictionary to save
    dict_path : Path
        Path to save the file

    Notes
    -----
    This rewrites the entire file. Use append_target_to_dict() for adding
    new targets without modifying existing content.
    """
    # Create backup
    backup_path = dict_path.with_suffix(".yaml.bak")
    if dict_path.exists():
        import shutil

        shutil.copy2(dict_path, backup_path)

    # Dump to string first
    yaml_str = yaml.dump(
        data,
        Dumper=_FlowStyleListDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )

    # Post-process to add blank lines between target entries
    # Pattern: line starting with exactly 2 spaces followed by a key (target name)
    import re

    lines = yaml_str.split("\n")
    output_lines = []

    for i, line in enumerate(lines):
        # Add blank line before each target entry (lines like "  01_PD1:")
        # but not for the first entry
        if i > 0 and re.match(r"^  [A-Za-z0-9_]+:", line) and not line.startswith("    "):
            # Check if previous line is not already blank
            if output_lines and output_lines[-1].strip() != "":
                output_lines.append("")
        output_lines.append(line)

    with open(dict_path, "w") as f:
        f.write("\n".join(output_lines))


def _format_target_entry(name: str, config: dict) -> str:
    """Format a single target entry as YAML string for appending.

    Parameters
    ----------
    name : str
        Target name
    config : dict
        Target configuration

    Returns
    -------
    str
        Formatted YAML string for this target entry
    """
    lines = [f"  {name}:"]

    for key, value in config.items():
        if value is None:
            lines.append(f"    {key}: null")
        elif isinstance(value, bool):
            lines.append(f"    {key}: {value!s}")
        elif isinstance(value, list):
            # Format list inline with quoted strings
            formatted_items = []
            for item in value:
                if item is None:
                    formatted_items.append("null")
                elif isinstance(item, str):
                    formatted_items.append(f'"{item}"')
                else:
                    formatted_items.append(str(item))
            lines.append(f"    {key}: [{', '.join(formatted_items)}]")
        elif isinstance(value, str):
            # Quote strings that need it
            import re

            chain_residue_pattern = re.compile(r"^[A-Z]\d+(-\d+)?$")
            needs_quotes = (
                any(
                    c in value
                    for c in [
                        ",",
                        ":",
                        "[",
                        "]",
                        "{",
                        "}",
                        "#",
                        "&",
                        "*",
                        "!",
                        "|",
                        ">",
                        "'",
                        "%",
                        "@",
                        "`",
                        "+",
                    ]
                )
                or chain_residue_pattern.match(value)
                or value == "LIGAND"
                or "-" in value
            )
            if needs_quotes:
                lines.append(f'    {key}: "{value}"')
            else:
                lines.append(f"    {key}: {value}")
        else:
            lines.append(f"    {key}: {value}")

    return "\n".join(lines)


def append_target_to_dict(name: str, config: dict, dict_path: Path) -> None:
    """Append a new target entry to the targets dict file.

    This preserves existing file content and just appends the new entry.
    Much faster and safer than rewriting the entire file.

    Parameters
    ----------
    name : str
        Target name
    config : dict
        Target configuration
    dict_path : Path
        Path to the targets dict file
    """
    # Format the new entry
    entry_str = _format_target_entry(name, config)

    # Read existing file to check if it ends with newline
    with open(dict_path) as f:
        content = f.read()

    # Ensure proper spacing before new entry
    if content.endswith("\n\n"):
        prefix = ""
    elif content.endswith("\n"):
        prefix = "\n"
    else:
        prefix = "\n\n"

    # Append new entry
    with open(dict_path, "a") as f:
        f.write(f"{prefix}{entry_str}\n")


def get_target_names(data: dict) -> list[str]:
    """Get list of target names from the targets dict.

    Parameters
    ----------
    data : dict
        The loaded targets dictionary

    Returns
    -------
    list[str]
        List of target names
    """
    if "target_dict_cfg" in data:
        return list(data["target_dict_cfg"].keys())
    return list(data.keys())


def format_target_entry(name: str, entry: dict, verbose: bool = False) -> str:
    """Format a target entry for display.

    Parameters
    ----------
    name : str
        Target name
    entry : dict
        Target configuration
    verbose : bool
        Whether to show all fields

    Returns
    -------
    str
        Formatted string
    """
    if verbose:
        lines = [f"  {name}:"]
        for key, value in entry.items():
            lines.append(f"    {key}: {value}")
        return "\n".join(lines)
    else:
        # Compact format: name - source/filename (input)
        source = entry.get("source", "?")
        filename = entry.get("target_filename", "?")
        is_ligand = "ligand" in entry
        target_path = entry.get("target_path")

        type_indicator = "🔬" if is_ligand else "🎯"

        if target_path:
            return f"  {type_indicator} {name:<30} path: {target_path}"
        else:
            detail = entry.get("ligand", entry.get("target_input", "?"))
            return f"  {type_indicator} {name:<30} {source}/{filename} ({detail})"


# =============================================================================
# List Command
# =============================================================================


def list_targets(
    dict_path: Path | None = None,
    verbose: bool = False,
    filter_ligand: bool | None = None,
) -> None:
    """List all targets in the targets dictionary.

    Parameters
    ----------
    dict_path : Path, optional
        Path to the targets dict file
    verbose : bool
        Show detailed information for each target
    filter_ligand : bool, optional
        If True, show only ligand targets. If False, show only protein targets.
        If None, show all targets.
    """
    try:
        data, resolved_path = load_targets_dict(dict_path)
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        sys.exit(1)

    targets = data.get("target_dict_cfg", data)
    target_names = list(targets.keys())

    # Filter if requested
    if filter_ligand is not None:
        target_names = [name for name in target_names if ("ligand" in targets[name]) == filter_ligand]

    # Count types
    n_ligand = sum(1 for name in targets if "ligand" in targets[name])
    n_protein = len(targets) - n_ligand

    print(f"\n{'═' * 60}")
    print("  🎯 Available Targets")
    print(f"{'═' * 60}")
    print(f"  Source: {resolved_path}")
    print(f"  Total:  {len(targets)} targets ({n_protein} protein, {n_ligand} ligand)")
    print(f"{'─' * 60}")

    if not target_names:
        print("  No targets found matching filter criteria.")
    else:
        for name in target_names:
            entry = targets[name]
            print(format_target_entry(name, entry, verbose=verbose))

    print(f"{'═' * 60}\n")


# =============================================================================
# Add Command - Interactive Mode
# =============================================================================


def create_target_template(
    name: str | None = None,
    defaults: dict | None = None,
    is_ligand: bool = False,
) -> str:
    """Create a YAML template for a new target entry.

    Parameters
    ----------
    name : str, optional
        Target name to use in template
    defaults : dict, optional
        Default values to populate
    is_ligand : bool, default False
        If True, include ligand-specific fields (ligand, ligand_only, SMILES,
        use_bonds_from_file). Ligand status is inferred from the presence of
        the ``ligand`` key at runtime — no explicit ``is_ligand`` flag needed.

    Returns
    -------
    str
        YAML template string
    """
    name = name or "new_target"
    defaults = defaults or {}

    target_type = "Ligand" if is_ligand else "Protein"

    # Common template lines with editor instructions header
    lines = [
        "# ═══════════════════════════════════════════════════════════════",
        f"# {target_type} Target Configuration Template",
        "# ═══════════════════════════════════════════════════════════════",
        "#",
        "# Instructions:",
        "#   SAVE and close    -> Add target",
        "#   CLOSE without save -> Cancel",
        "#",
        "# vim:          :wq (save+quit)  |  :q! (cancel)  |  i (edit)  |  Esc",
        "# nano:         Ctrl+O (save)    |  Ctrl+X (exit)",
        "# code/cursor:  Ctrl+S (save)    |  Close tab",
        "#",
        "# ═══════════════════════════════════════════════════════════════",
        "#",
        "# Required: EITHER (source + target_filename) OR target_path",
        "# For protein targets: target_input (chain/residue range) is required.",
        "# For ligand targets: the 'ligand' field determines ligand status.",
        "#",
        "# If you have a full path to your PDB, just set target_path.",
        "# Otherwise, set source and target_filename to build the path from DATA_PATH.",
        "",
        f"name: {name}",
        "",
        "# Option 1: Source directory under $DATA_PATH/target_data/",
        "# (Required if target_path is not set)",
        f"source: {defaults.get('source', 'ligand_targets' if is_ligand else 'custom_targets')}",
        "",
        "# Target PDB filename without .pdb extension (Required if target_path is not set)",
        f"target_filename: {defaults.get('target_filename', name)}",
        "",
        "# Option 2: Full path to target PDB (if set, source/target_filename are ignored)",
        f"target_path: {defaults.get('target_path', '')}",
        "",
    ]

    if not is_ligand:
        lines.extend(
            [
                "# Chain and residue range (e.g., 'A1-115' for chain A, residues 1-115)",
                f"target_input: {defaults.get('target_input', 'A1-100')}",
                "",
            ]
        )

    lines.extend(
        [
            "# Hotspot residues - key interface residues to focus on",
            "# Format: list like ['A33', 'A95'] or leave empty []",
            f"hotspot_residues: {defaults.get('hotspot_residues', [])}",
            "",
            "# Binder length range [min, max] or single value [length]",
            f"binder_length: {defaults.get('binder_length', [60, 120])}",
            "",
            "# Reference PDB ID (optional)",
            f"pdb_id: {defaults.get('pdb_id', 'null')}",
        ]
    )

    if is_ligand:
        lines.extend(
            [
                "",
                "# ═══════════════════════════════════════════════════════════════",
                "# LIGAND-SPECIFIC CONFIGURATION",
                "# ═══════════════════════════════════════════════════════════════",
                "",
                "# Ligand residue name(s) from PDB (e.g., 'FAD', 'OQO', ['DHZ', 'ZN'])",
                "# Presence of this field marks the target as a ligand target.",
                f"ligand: '{defaults.get('ligand', 'UNK')}'",
                "",
                "# If True, use the entire file as the ligand (no extraction by residue name)",
                f"ligand_only: {defaults.get('ligand_only', True)!s}",
                "",
                "# SMILES string for the ligand molecule (single-ligand only)",
                f'SMILES: "{defaults.get("SMILES", "")}"',
                "",
                "# If True, use bond information from the input PDB/CIF file",
                f"use_bonds_from_file: {defaults.get('use_bonds_from_file', True)!s}",
            ]
        )

    return "\n".join(lines)


def parse_target_template(content: str) -> tuple[str, dict]:
    """Parse a target template string into name and config.

    Parameters
    ----------
    content : str
        YAML template content

    Returns
    -------
    tuple[str, dict]
        Target name and configuration dictionary
    """
    # Parse YAML
    data = yaml.safe_load(content)

    if not data:
        raise ValueError("Empty or invalid template")

    # Extract name
    name = data.pop("name", None)
    if not name:
        raise ValueError("Target name is required")

    # Process fields
    config = {}

    # Boolean fields that should be parsed as True/False
    boolean_fields = {"ligand_only", "use_bonds_from_file"}
    # Fields that can be null/None
    nullable_fields = {"pdb_id", "ligand", "target_path", "SMILES"}

    for field, value in data.items():
        if field not in TARGET_FIELDS:
            print(f"  ⚠️  Unknown field '{field}' will be included anyway")

        # Handle null/None values
        if value in (None, "null", "None", ""):
            if field in nullable_fields:
                config[field] = None
            continue

        # Handle boolean fields
        if field in boolean_fields:
            if isinstance(value, str):
                config[field] = value.lower() in ("true", "yes", "1")
            else:
                config[field] = bool(value)
            continue

        # Handle lists (hotspot_residues, binder_length)
        if field in ["hotspot_residues", "binder_length"]:
            if isinstance(value, str):
                # Try to parse as YAML list
                try:
                    value = yaml.safe_load(value)
                except:
                    pass
            if isinstance(value, list):
                config[field] = value
            else:
                config[field] = [value]
            continue

        config[field] = value

    # Validate required fields
    # target_input is always required
    missing_always = [f for f in REQUIRED_FIELDS_ALWAYS if f not in config or not config[f]]
    if missing_always:
        raise ValueError(f"Missing required fields: {', '.join(missing_always)}")

    # Either target_path OR (source AND target_filename) is required
    has_target_path = config.get("target_path") is not None and config.get("target_path") != ""
    has_source_filename = (
        config.get("source") is not None
        and config.get("source") != ""
        and config.get("target_filename") is not None
        and config.get("target_filename") != ""
    )

    if not has_target_path and not has_source_filename:
        raise ValueError("Either 'target_path' OR both 'source' and 'target_filename' must be provided")

    return name, config


def get_editor(preferred: str | None = None) -> str:
    """Get the preferred text editor.

    Parameters
    ----------
    preferred : str, optional
        User-specified editor. If provided and available, use it.

    Returns
    -------
    str
        Editor command to use
    """
    import shutil

    # If user specified an editor, try to use it
    if preferred:
        # Handle common editor shortcuts
        editor_aliases = {
            "code": "code --wait",  # VS Code needs --wait to block
            "vscode": "code --wait",
            "cursor": "cursor --wait",  # Cursor IDE needs --wait to block
            "subl": "subl --wait",  # Sublime needs --wait
            "sublime": "subl --wait",
            "atom": "atom --wait",
            "gedit": "gedit",
            "notepad++": "notepad++",
            "nano": "nano",
            "vim": "vim",
            "vi": "vi",
            "emacs": "emacs",
            "nvim": "nvim",
            "neovim": "nvim",
        }

        # Check if it's an alias
        if preferred.lower() in editor_aliases:
            return editor_aliases[preferred.lower()]

        # Otherwise use as-is (user knows what they're doing)
        return preferred

    # Check environment variables
    for env_var in ["VISUAL", "EDITOR"]:
        editor = os.environ.get(env_var)
        if editor:
            return editor

    # Try common editors (prefer nano over vim for beginners)
    for editor in ["nano", "vim", "vi", "notepad"]:
        if shutil.which(editor):
            return editor

    return "nano"  # Fallback


def add_target_interactive(
    dict_path: Path | None = None,
    name: str | None = None,
    defaults: dict | None = None,
    is_ligand: bool = False,
    editor: str | None = None,
) -> bool:
    """Add a target using an interactive editor.

    Parameters
    ----------
    dict_path : Path, optional
        Path to the targets dict file
    name : str, optional
        Initial target name
    defaults : dict, optional
        Default values to populate
    is_ligand : bool, default False
        If True, use ligand-specific template with ligand, SMILES, etc.
    editor : str, optional
        Editor to use. If None, uses VISUAL/EDITOR env var or defaults to nano.
        Supports shortcuts: 'code', 'vscode', 'subl', 'nano', 'vim', etc.

    Returns
    -------
    bool
        True if target was added successfully
    """
    try:
        data, resolved_path = load_targets_dict(dict_path)
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        return False

    # Create template (ligand vs protein)
    template = create_target_template(name, defaults, is_ligand=is_ligand)

    # Create temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="complexa_target_",
        delete=False,
    ) as f:
        f.write(template)
        temp_path = f.name

    try:
        # Open editor
        editor_cmd = get_editor(editor)

        # Show relative path if possible, otherwise absolute
        try:
            display_path = resolved_path.relative_to(Path.cwd())
        except ValueError:
            display_path = resolved_path

        print(f"\n  Opening editor ({editor_cmd})...")
        print(f"  📍 Target dict: {display_path}")
        print("  Edit the template and save to add the target.")
        print("  Close the editor without saving to cancel.\n")

        # Run editor (use shell=True to handle commands with arguments like "code --wait")
        result = subprocess.run(f"{editor_cmd} {temp_path}", shell=True)

        if result.returncode != 0:
            print("\n  ✗ Editor exited with error")
            return False

        # Read edited content
        with open(temp_path) as f:
            content = f.read()

        # Check if content was modified
        if content.strip() == template.strip():
            print("\n  ✗ No changes made, cancelled")
            return False

        # Parse template
        try:
            target_name, config = parse_target_template(content)
        except ValueError as e:
            print(f"\n  ✗ Invalid template: {e}")
            return False

        # Check if target already exists
        targets = data.get("target_dict_cfg", data)
        target_exists = target_name in targets

        if target_exists:
            print(f"\n  ⚠️  Target '{target_name}' already exists!")
            try:
                resp = input("  Overwrite? (y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.")
                return False
            if resp != "y":
                print("  Cancelled.")
                return False

            # Overwrite existing - need to rewrite file
            if "target_dict_cfg" in data:
                data["target_dict_cfg"][target_name] = config
            else:
                data[target_name] = config
            save_targets_dict(data, resolved_path)
            print(f"\n  ✓ Updated target '{target_name}'")
        else:
            # New target - just append to file (preserves existing content)
            append_target_to_dict(target_name, config, resolved_path)
            print(f"\n  ✓ Added target '{target_name}'")
        print(f"  📍 Saved to: {display_path}")
        print("\n  Configuration:")
        for key, value in config.items():
            print(f"    {key}: {value}")

        return True

    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass


# =============================================================================
# Add Command - Non-Interactive Mode
# =============================================================================


def add_target_cli(
    name: str,
    dict_path: Path | None = None,
    source: str | None = None,
    target_filename: str | None = None,
    target_path: str | None = None,
    target_input: str | None = None,
    hotspot_residues: list[str] | None = None,
    binder_length: list[int] | None = None,
    pdb_id: str | None = None,
    ligand: str | None = None,
    ligand_only: bool | None = None,
    smiles: str | None = None,
    use_bonds_from_file: bool | None = None,
    force: bool = False,
) -> bool:
    """Add a target using command-line arguments.

    Parameters
    ----------
    name : str
        Target name (required)
    dict_path : Path, optional
        Path to the targets dict file
    source : str, optional
        Source directory (default: 'custom_targets' or 'ligand_targets' for ligands)
    target_filename : str, optional
        Target filename (default: same as name)
    target_path : str, optional
        Full path to target PDB (overrides source/target_filename)
    target_input : str, optional
        Chain and residue range (default: 'A1-100', protein targets only)
    hotspot_residues : list[str], optional
        List of hotspot residues
    binder_length : list[int], optional
        Binder length range (default: [60, 120])
    pdb_id : str, optional
        Reference PDB ID
    ligand : str, optional
        Ligand residue name(s). Presence of this field marks the target as a
        ligand target.
    ligand_only : bool, optional
        If True, use entire file as ligand (for ligand targets)
    smiles : str, optional
        SMILES string for the ligand molecule (for ligand targets)
    use_bonds_from_file : bool, optional
        If True, use bond information from the input PDB file (for ligand targets)
    force : bool
        Overwrite existing target without prompting

    Returns
    -------
    bool
        True if target was added successfully
    """
    try:
        data, resolved_path = load_targets_dict(dict_path)
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        return False

    # Show relative path if possible
    try:
        display_path = resolved_path.relative_to(Path.cwd())
    except ValueError:
        display_path = resolved_path

    # Validate: Either target_path OR (source/target_filename) is required
    has_target_path = target_path is not None and target_path != ""
    is_ligand = ligand is not None

    # Build config
    config = {}
    if not is_ligand:
        config["target_input"] = target_input or "A1-100"

    # Default source depends on ligand vs protein
    default_source = "ligand_targets" if is_ligand else "custom_targets"

    if has_target_path:
        # If target_path is provided, use it and source/target_filename are optional
        config["target_path"] = target_path
        if source:
            config["source"] = source
        if target_filename:
            config["target_filename"] = target_filename
    else:
        # If no target_path, source and target_filename are required (use defaults)
        config["source"] = source or default_source
        config["target_filename"] = target_filename or name

    # Add optional fields if provided
    if hotspot_residues:
        config["hotspot_residues"] = hotspot_residues
    else:
        config["hotspot_residues"] = []
    if binder_length:
        config["binder_length"] = binder_length
    else:
        config["binder_length"] = [60, 120]
    if pdb_id:
        config["pdb_id"] = pdb_id
    else:
        config["pdb_id"] = None

    # Ligand-specific fields (presence of 'ligand' key marks this as a ligand target)
    if is_ligand:
        config["ligand"] = ligand
        if ligand_only is not None:
            config["ligand_only"] = ligand_only
        else:
            config["ligand_only"] = True
        if smiles:
            config["SMILES"] = smiles
        if use_bonds_from_file is not None:
            config["use_bonds_from_file"] = use_bonds_from_file
        else:
            config["use_bonds_from_file"] = True

    # Check if target already exists
    targets = data.get("target_dict_cfg", data)
    target_exists = name in targets

    if target_exists and not force:
        print(f"\n  ⚠️  Target '{name}' already exists!")
        try:
            resp = input("  Overwrite? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return False
        if resp != "y":
            print("  Cancelled.")
            return False

    if target_exists:
        # Overwrite existing - need to rewrite file
        if "target_dict_cfg" in data:
            data["target_dict_cfg"][name] = config
        else:
            data[name] = config
        save_targets_dict(data, resolved_path)
        print(f"\n  ✓ Updated target '{name}'")
    else:
        # New target - just append to file (preserves existing content)
        append_target_to_dict(name, config, resolved_path)
        print(f"\n  ✓ Added target '{name}'")
    print(f"  📍 Saved to: {display_path}")
    print("\n  Configuration:")
    for key, value in config.items():
        print(f"    {key}: {value}")

    return True


# =============================================================================
# Show Command
# =============================================================================


def show_target(
    name: str,
    dict_path: Path | None = None,
) -> None:
    """Show detailed information about a specific target.

    Parameters
    ----------
    name : str
        Target name
    dict_path : Path, optional
        Path to the targets dict file
    """
    try:
        data, resolved_path = load_targets_dict(dict_path)
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        sys.exit(1)

    targets = data.get("target_dict_cfg", data)

    if name not in targets:
        print(f"\n  ✗ Target '{name}' not found")
        print("\n  Available targets:")
        for n in list(targets.keys())[:10]:
            print(f"    - {n}")
        if len(targets) > 10:
            print(f"    ... and {len(targets) - 10} more")
        sys.exit(1)

    entry = targets[name]

    print(f"\n{'═' * 60}")
    print(f"  🎯 Target: {name}")
    print(f"{'═' * 60}")
    print(f"  Source: {resolved_path}")
    print(f"{'─' * 60}")

    for key, value in entry.items():
        print(f"  {key:<20} : {value}")

    print(f"{'═' * 60}\n")


# =============================================================================
# Main Entry Point (for testing)
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Target management CLI")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list", help="List targets")
    list_parser.add_argument("--dict", type=Path, help="Path to targets dict")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show details")
    list_parser.add_argument("--ligand", action="store_true", help="Show only ligand targets")
    list_parser.add_argument("--protein", action="store_true", help="Show only protein targets")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a target")
    add_parser.add_argument("name", nargs="?", help="Target name")
    add_parser.add_argument("--dict", type=Path, help="Path to targets dict")
    add_parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    add_parser.add_argument("--source", help="Source directory")
    add_parser.add_argument("--target-filename", help="Target filename")
    add_parser.add_argument("--target-path", help="Full path to target PDB")
    add_parser.add_argument("--target-input", help="Chain and residue range")
    add_parser.add_argument("--hotspot-residues", nargs="+", help="Hotspot residues")
    add_parser.add_argument("--binder-length", type=int, nargs="+", help="Binder length range")
    add_parser.add_argument("--pdb-id", help="Reference PDB ID")
    add_parser.add_argument("--ligand", help="Ligand residue name(s) — marks target as ligand")
    add_parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show target details")
    show_parser.add_argument("name", help="Target name")
    show_parser.add_argument("--dict", type=Path, help="Path to targets dict")

    args = parser.parse_args()

    if args.command == "list":
        filter_ligand = None
        if args.ligand:
            filter_ligand = True
        elif args.protein:
            filter_ligand = False
        list_targets(args.dict, verbose=args.verbose, filter_ligand=filter_ligand)
    elif args.command == "add":
        if args.interactive or not args.name:
            add_target_interactive(args.dict, args.name)
        else:
            add_target_cli(
                name=args.name,
                dict_path=args.dict,
                source=args.source,
                target_filename=args.target_filename,
                target_path=args.target_path,
                target_input=args.target_input,
                hotspot_residues=args.hotspot_residues,
                binder_length=args.binder_length,
                pdb_id=args.pdb_id,
                ligand=args.ligand,
                force=args.force,
            )
    elif args.command == "show":
        show_target(args.name, args.dict)
    else:
        parser.print_help()
