#!/usr/bin/env python3
"""
Standalone target management CLI - minimal imports for fast startup.

This is a lightweight entry point for target commands only.
Usage: complexa-target list|add|show [options]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="complexa-target",
        description="Manage target configurations for binder design",
    )
    subparsers = parser.add_subparsers(dest="command")

    # list
    list_parser = subparsers.add_parser("list", help="List available targets")
    list_parser.add_argument("--dict", type=Path, help="Path to custom targets dict")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show details")
    list_parser.add_argument("--ligand", action="store_true", help="Only ligand targets")
    list_parser.add_argument("--protein", action="store_true", help="Only protein targets")

    # show
    show_parser = subparsers.add_parser("show", help="Show target details")
    show_parser.add_argument("name", help="Target name")
    show_parser.add_argument("--dict", type=Path, help="Path to custom targets dict")

    # add
    add_parser = subparsers.add_parser("add", help="Add a new target")
    add_parser.add_argument("name", nargs="?", help="Target name")
    add_parser.add_argument("--dict", type=Path, help="Path to custom targets dict")
    add_parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    add_parser.add_argument("-e", "--editor", type=str, help="Editor (e.g., 'code', 'nano', 'vim')")
    add_parser.add_argument("--source", type=str, help="Source directory")
    add_parser.add_argument("--target-filename", type=str, help="Target PDB filename")
    add_parser.add_argument("--target-path", type=str, help="Full path to target PDB")
    add_parser.add_argument("--target-input", type=str, help="Chain and residue range")
    add_parser.add_argument("--hotspot-residues", nargs="+", help="Hotspot residues")
    add_parser.add_argument("--binder-length", type=int, nargs="+", help="Binder length range")
    add_parser.add_argument("--pdb-id", type=str, help="Reference PDB ID")
    add_parser.add_argument(
        "--ligand",
        type=str,
        nargs="?",
        const="YOUR_LIGAND",
        help="Ligand residue name(s) — marks target as ligand (e.g., 'FAD', 'OQO'). Use without a value in interactive mode to get a placeholder.",
    )
    add_parser.add_argument(
        "--ligand-only",
        action="store_true",
        default=None,
        help="Generate pocket around ligand only",
    )
    add_parser.add_argument("--smiles", type=str, help="SMILES string for ligand molecule")
    add_parser.add_argument(
        "--use-bonds-from-file",
        action="store_true",
        default=None,
        help="Use bonds from PDB file",
    )
    add_parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Lazy import target_manager only after parsing args
    from proteinfoundation.cli.target_manager import add_target_cli, add_target_interactive, list_targets, show_target

    if args.command == "list":
        filter_ligand = None
        if args.ligand:
            filter_ligand = True
        elif args.protein:
            filter_ligand = False
        list_targets(dict_path=args.dict, verbose=args.verbose, filter_ligand=filter_ligand)

    elif args.command == "show":
        show_target(name=args.name, dict_path=args.dict)

    elif args.command == "add":
        is_ligand = getattr(args, "ligand", None) is not None

        if args.interactive or not args.name:
            defaults = {}
            for key in [
                "source",
                "target_filename",
                "target_path",
                "target_input",
                "hotspot_residues",
                "binder_length",
                "pdb_id",
                "ligand",
            ]:
                val = getattr(args, key.replace("-", "_"), None)
                if val:
                    defaults[key] = val
            if getattr(args, "smiles", None):
                defaults["SMILES"] = args.smiles
            if getattr(args, "ligand_only", None) is not None:
                defaults["ligand_only"] = args.ligand_only
            if getattr(args, "use_bonds_from_file", None) is not None:
                defaults["use_bonds_from_file"] = args.use_bonds_from_file

            success = add_target_interactive(
                dict_path=args.dict,
                name=args.name,
                defaults=defaults or None,
                is_ligand=is_ligand,
                editor=getattr(args, "editor", None),
            )
        else:
            success = add_target_cli(
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
                ligand_only=getattr(args, "ligand_only", None),
                smiles=getattr(args, "smiles", None),
                use_bonds_from_file=getattr(args, "use_bonds_from_file", None),
                force=args.force,
            )
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
